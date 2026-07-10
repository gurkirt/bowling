//
//  HighlightBuilder.swift
//  CricReel
//
//  Stitches per-ball clips into highlight reels with a two-line delivery overlay burned
//  into the top of each clip. All rendering runs off the scoring path (async).
//

import Foundation
import AVFoundation
import UIKit
import Combine

/// A clip plus the caption content to burn onto it.
struct ReelClip {
    let url: URL
    let info: OverlayInfo
}

@MainActor
final class HighlightBuilder: ObservableObject {
    static let shared = HighlightBuilder()

    @Published var isBuilding = false
    @Published var statusMessage: String?
    @Published var lastReelURL: URL?

    private init() {}

    /// Auto reel for a completed over (its 4s/6s/wickets).
    func buildOverReel(match: Match, innings: Innings, overNumber: Int, lookup: PlayerLookup) {
        let clips = reelClips(from: innings.orderedDeliveries.filter {
            $0.overNumber == overNumber && !$0.highlightTags.isEmpty
        }, match: match, lookup: lookup)
        guard !clips.isEmpty else { return }
        let out = ClipStore.highlightsDirectory
            .appendingPathComponent("over_\(innings.order)_\(overNumber + 1)_\(shortID()).mp4")
        render(clips: clips, output: out, doneMessage: "Over \(overNumber + 1) reel ready")
    }

    /// Build a reel from explicit clips (gathered by the caller). Returns the URL, or nil.
    func buildReel(clips: [ReelClip], name: String) async -> URL? {
        guard !clips.isEmpty else { return nil }
        let out = ClipStore.highlightsDirectory.appendingPathComponent("\(name)_\(shortID()).mp4")
        isBuilding = true
        statusMessage = "Building reel…"
        do {
            let result = try await Self.stitch(clips: clips, outputURL: out)
            isBuilding = false; statusMessage = "Reel ready"; lastReelURL = result
            return result
        } catch {
            isBuilding = false; statusMessage = "Reel failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// Map deliveries (that have an existing clip) to ReelClips with caption content.
    func reelClips(from deliveries: [Delivery], match: Match, lookup: PlayerLookup) -> [ReelClip] {
        deliveries.compactMap { d in
            guard let name = d.clipFilename, ClipStore.clipExists(name) else { return nil }
            return ReelClip(url: ClipStore.url(forClip: name),
                            info: ClipOverlay.info(for: d, match: match, lookup: lookup))
        }
    }

    /// Export a single clip with the caption burned in, to a temporary file for sharing.
    /// The stored clip stays raw; the caption is only baked here, on demand.
    func exportCaptionedClip(_ clip: ReelClip) async -> URL? {
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("cricreel_share_\(shortID()).mp4")
        return try? await Self.stitch(clips: [clip], outputURL: out)
    }

    // MARK: - Internal

    private func render(clips: [ReelClip], output: URL, doneMessage: String) {
        isBuilding = true
        statusMessage = "Building reel…"
        Task {
            do {
                let result = try await Self.stitch(clips: clips, outputURL: output)
                self.lastReelURL = result
                self.statusMessage = doneMessage
            } catch {
                self.statusMessage = "Reel failed: \(error.localizedDescription)"
            }
            self.isBuilding = false
        }
    }

    private func shortID() -> String { String(UUID().uuidString.prefix(6)) }

    // MARK: - Stitching + overlay burn-in

    nonisolated static func stitch(clips: [ReelClip], outputURL: URL) async throws -> URL {
        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(
            withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw ReelError.trackCreationFailed
        }

        var cursor = CMTime.zero
        var renderSize = CGSize(width: 1080, height: 1920)
        var preferred = CGAffineTransform.identity
        var segments: [(start: CMTime, duration: CMTime, clip: ReelClip)] = []

        for clip in clips {
            let asset = AVURLAsset(url: clip.url)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let src = tracks.first else { continue }
            let duration = try await asset.load(.duration)
            try videoTrack.insertTimeRange(CMTimeRange(start: .zero, duration: duration),
                                           of: src, at: cursor)
            let naturalSize = try await src.load(.naturalSize)
            preferred = try await src.load(.preferredTransform)
            let rect = CGRect(origin: .zero, size: naturalSize).applying(preferred)
            renderSize = CGSize(width: abs(rect.width), height: abs(rect.height))
            segments.append((cursor, duration, clip))
            cursor = CMTimeAdd(cursor, duration)
        }
        guard cursor > .zero else { throw ReelError.noUsableClips }
        videoTrack.preferredTransform = preferred

        // Reels are for sharing: cap the export at HD (1080 short side) and 30 fps.
        // 4K source clips are scaled down; HD clips pass through unchanged.
        let downScale = min(1.0,
                            1080.0 / min(renderSize.width, renderSize.height),
                            1920.0 / max(renderSize.width, renderSize.height))
        if downScale < 1.0 {
            let even = { (v: CGFloat) in (v / 2).rounded(.down) * 2 }
            renderSize = CGSize(width: even(renderSize.width * downScale),
                                height: even(renderSize.height * downScale))
        }

        let total = cursor
        let (parentLayer, videoLayer) = await Self.buildLayers(renderSize: renderSize,
                                                               segments: segments, total: total)

        let videoComposition = AVMutableVideoComposition()
        videoComposition.renderSize = renderSize
        videoComposition.frameDuration = CMTime(value: 1, timescale: 30)
        let instruction = AVMutableVideoCompositionInstruction()
        instruction.timeRange = CMTimeRange(start: .zero, duration: total)
        let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)
        layerInstruction.setTransform(
            preferred.concatenating(CGAffineTransform(scaleX: downScale, y: downScale)), at: .zero)
        instruction.layerInstructions = [layerInstruction]
        videoComposition.instructions = [instruction]
        videoComposition.animationTool = AVVideoCompositionCoreAnimationTool(
            postProcessingAsVideoLayer: videoLayer, in: parentLayer)

        try? FileManager.default.removeItem(at: outputURL)
        guard let export = AVAssetExportSession(
            asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw ReelError.exportSetupFailed
        }
        export.outputURL = outputURL
        export.outputFileType = .mp4
        export.videoComposition = videoComposition
        export.shouldOptimizeForNetworkUse = true

        return try await withCheckedThrowingContinuation { continuation in
            export.exportAsynchronously {
                switch export.status {
                case .completed: continuation.resume(returning: outputURL)
                case .failed, .cancelled: continuation.resume(throwing: export.error ?? ReelError.exportFailed)
                default: continuation.resume(throwing: ReelError.exportFailed)
                }
            }
        }
    }

    /// Build the parent/video layers plus a timed lower-third caption per segment.
    @MainActor
    private static func buildLayers(renderSize: CGSize,
                                    segments: [(start: CMTime, duration: CMTime, clip: ReelClip)],
                                    total: CMTime) -> (CALayer, CALayer) {
        let parentLayer = CALayer()
        let videoLayer = CALayer()
        parentLayer.frame = CGRect(origin: .zero, size: renderSize)
        videoLayer.frame = parentLayer.frame
        parentLayer.addSublayer(videoLayer)

        let totalSeconds = CMTimeGetSeconds(total)
        for seg in segments {
            let ss = CMTimeGetSeconds(seg.start)
            let ee = ss + CMTimeGetSeconds(seg.duration)
            let card = makeCard(seg.clip.info, renderSize: renderSize,
                                segStart: ss, segEnd: ee, total: totalSeconds)
            parentLayer.addSublayer(card)
        }
        return (parentLayer, videoLayer)
    }

    /// A caption card near the TOP (matching in-app playback). Score row flips from
    /// before-this-ball to after-this-ball; only the outcome row animates in.
    @MainActor
    private static func makeCard(_ info: OverlayInfo, renderSize: CGSize,
                                 segStart: Double, segEnd: Double, total: Double) -> CALayer {
        let W = renderSize.width, H = renderSize.height
        let cardW = W * 0.66, cardH = H * 0.155
        let cardX = (W - cardW) / 2

        let card = CALayer()
        // CoreAnimation origin is bottom-left → high y = visually near the top; centered.
        card.frame = CGRect(x: cardX, y: H * 0.80, width: cardW, height: cardH)
        card.backgroundColor = UIColor.black.withAlphaComponent(0.55).cgColor
        card.cornerRadius = H * 0.012
        card.masksToBounds = true
        card.opacity = 0

        // Accent bar down the left edge — colour-codes fours/sixes/wickets like the app UI.
        let accentBar = CALayer()
        accentBar.frame = CGRect(x: 0, y: 0, width: max(3, cardW * 0.012), height: cardH)
        accentBar.backgroundColor = info.accent.uiColor.cgColor
        card.addSublayer(accentBar)

        let textX: CGFloat = 10
        let textW = cardW - 20

        // Row 3 (outcome) — the only animated row.
        let outcome = textLayer(info.outcome, size: H * 0.030, weight: .heavy, color: info.accent.uiColor)
        outcome.alignmentMode = .center
        outcome.frame = CGRect(x: textX, y: cardH * 0.06, width: textW, height: cardH * 0.36)
        card.addSublayer(outcome)

        // Row 2 (delivery) — static.
        let delivery = textLayer(info.delivery, size: H * 0.022, weight: .medium, color: .white)
        delivery.alignmentMode = .center
        delivery.frame = CGRect(x: textX, y: cardH * 0.42, width: textW, height: cardH * 0.24)
        card.addSublayer(delivery)

        // Row 1 (score) — static position, flips before → after in place.
        let scoreFrame = CGRect(x: textX, y: cardH * 0.66, width: textW, height: cardH * 0.30)
        let scoreBefore = textLayer(info.scoreBefore, size: H * 0.024, weight: .semibold, color: .white)
        let scoreAfter = textLayer(info.scoreAfter, size: H * 0.024, weight: .semibold, color: .white)
        scoreBefore.alignmentMode = .center
        scoreAfter.alignmentMode = .center
        scoreBefore.frame = scoreFrame
        scoreAfter.frame = scoreFrame
        scoreAfter.opacity = 0
        card.addSublayer(scoreBefore)
        card.addSublayer(scoreAfter)

        func n(_ t: Double) -> Double { max(0, min(1, t / total)) }
        let s = n(segStart), e = n(segEnd)
        let flip = n(min(segStart + 1.0, segEnd - 0.05))
        let reveal = n(segStart + 0.35)

        // Card visible for the whole segment (hard cut in/out) — reliable in this pipeline.
        card.add(opacityKeyframe([0, 0, 1, 1, 0, 0], [0, s, s, e, e, 1], total, .discrete), forKey: "vis")

        // Score flip (same discrete technique as the visibility → reliable).
        scoreBefore.add(opacityKeyframe([0, 0, 1, 1, 0, 0], [0, s, s, flip, flip, 1], total, .discrete), forKey: "sb")
        scoreAfter.add(opacityKeyframe([0, 0, 1, 1, 0, 0], [0, flip, flip, e, e, 1], total, .discrete), forKey: "sa")

        // Row 3 entrance: fade + slide in.
        outcome.add(opacityKeyframe([0, 0, 1, 1, 0, 0], [0, s, reveal, e, e, 1], total, .linear), forKey: "oo")
        let restX = outcome.position.x
        let slide = CAKeyframeAnimation(keyPath: "position.x")
        slide.values = [restX - cardW * 0.10, restX - cardW * 0.10, restX, restX]
        slide.keyTimes = [0, s, reveal, 1].map { NSNumber(value: $0) }
        slide.calculationMode = .linear
        slide.beginTime = AVCoreAnimationBeginTimeAtZero
        slide.duration = total
        slide.isRemovedOnCompletion = false
        slide.fillMode = .both
        outcome.add(slide, forKey: "os")

        return card
    }

    private static func opacityKeyframe(_ values: [Double], _ keyTimes: [Double],
                                        _ total: Double, _ mode: CAAnimationCalculationMode) -> CAKeyframeAnimation {
        let a = CAKeyframeAnimation(keyPath: "opacity")
        a.values = values
        a.keyTimes = keyTimes.map { NSNumber(value: max(0, min(1, $0))) }
        a.calculationMode = mode
        a.beginTime = AVCoreAnimationBeginTimeAtZero
        a.duration = total
        a.isRemovedOnCompletion = false
        a.fillMode = .both
        return a
    }

    @MainActor
    private static func textLayer(_ string: String, size: CGFloat,
                                  weight: UIFont.Weight, color: UIColor) -> CATextLayer {
        let layer = CATextLayer()
        layer.contentsScale = 2
        layer.alignmentMode = .left
        layer.isWrapped = false
        layer.truncationMode = .end
        let shadow = NSShadow()
        shadow.shadowColor = UIColor.black.withAlphaComponent(0.9)
        shadow.shadowBlurRadius = 4
        shadow.shadowOffset = CGSize(width: 0, height: 1)
        layer.string = NSAttributedString(string: string, attributes: [
            .font: UIFont.systemFont(ofSize: size, weight: weight),
            .foregroundColor: color,
            .strokeColor: UIColor.black, .strokeWidth: -2.0,
            .shadow: shadow])
        return layer
    }

    enum ReelError: LocalizedError {
        case trackCreationFailed, noUsableClips, exportSetupFailed, exportFailed
        var errorDescription: String? {
            switch self {
            case .trackCreationFailed: return "Could not create video track."
            case .noUsableClips:       return "No usable clips to stitch."
            case .exportSetupFailed:   return "Could not set up export."
            case .exportFailed:        return "Export failed."
            }
        }
    }
}
