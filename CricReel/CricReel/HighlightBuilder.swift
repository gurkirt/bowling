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

/// A clip plus the overlay text to burn onto it.
struct ReelClip {
    let url: URL
    let line1: String
    let line2: String
    let line3: String
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

    /// Map deliveries (that have an existing clip) to ReelClips with overlay text.
    func reelClips(from deliveries: [Delivery], match: Match, lookup: PlayerLookup) -> [ReelClip] {
        deliveries.compactMap { d in
            guard let name = d.clipFilename, ClipStore.clipExists(name) else { return nil }
            let (l1, l2, l3) = DeliveryFormatting.overlayLines(d, match: match, lookup: lookup)
            return ReelClip(url: ClipStore.url(forClip: name), line1: l1, line2: l2, line3: l3)
        }
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

        let total = cursor
        let (parentLayer, videoLayer) = await Self.buildLayers(renderSize: renderSize,
                                                               segments: segments, total: total)

        let videoComposition = AVMutableVideoComposition()
        videoComposition.renderSize = renderSize
        videoComposition.frameDuration = CMTime(value: 1, timescale: 30)
        let instruction = AVMutableVideoCompositionInstruction()
        instruction.timeRange = CMTimeRange(start: .zero, duration: total)
        let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)
        layerInstruction.setTransform(preferred, at: .zero)
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

    /// Build the parent/video layers plus a timed text overlay for each segment.
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
            let text = Self.makeTextLayer(seg.clip, renderSize: renderSize)
            // Origin is bottom-left for the animation tool → place near the top (10–30%).
            text.frame = CGRect(x: renderSize.width * 0.04, y: renderSize.height * 0.64,
                                width: renderSize.width * 0.92, height: renderSize.height * 0.24)
            text.opacity = 0
            let s = max(0, min(1, CMTimeGetSeconds(seg.start) / totalSeconds))
            let e = max(s, min(1, (CMTimeGetSeconds(seg.start) + CMTimeGetSeconds(seg.duration)) / totalSeconds))
            let anim = CAKeyframeAnimation(keyPath: "opacity")
            // Visible only within [s, e]; keyTimes span the full [0,1] so segments never bleed.
            anim.values = [0, 0, 1, 1, 0, 0]
            anim.keyTimes = [0, NSNumber(value: s), NSNumber(value: s),
                             NSNumber(value: e), NSNumber(value: e), 1]
            anim.calculationMode = .discrete
            anim.beginTime = AVCoreAnimationBeginTimeAtZero
            anim.duration = totalSeconds
            anim.isRemovedOnCompletion = false
            anim.fillMode = .both
            text.add(anim, forKey: "opacity")
            parentLayer.addSublayer(text)
        }
        return (parentLayer, videoLayer)
    }

    @MainActor
    private static func makeTextLayer(_ clip: ReelClip, renderSize: CGSize) -> CATextLayer {
        let layer = CATextLayer()
        layer.contentsScale = 2
        layer.alignmentMode = .center
        layer.isWrapped = true
        layer.truncationMode = .end
        // Dark translucent band for contrast against bright pitches / sky.
        layer.backgroundColor = UIColor.black.withAlphaComponent(0.38).cgColor
        layer.cornerRadius = renderSize.height * 0.012
        layer.masksToBounds = true

        let f1 = UIFont.systemFont(ofSize: renderSize.height * 0.026, weight: .semibold) // scores
        let f2 = UIFont.systemFont(ofSize: renderSize.height * 0.024, weight: .medium)   // delivery
        let f3 = UIFont.systemFont(ofSize: renderSize.height * 0.032, weight: .heavy)     // outcome
        let shadow = NSShadow()
        shadow.shadowColor = UIColor.black.withAlphaComponent(0.9)
        shadow.shadowBlurRadius = 5
        shadow.shadowOffset = CGSize(width: 0, height: 1)

        func attrs(_ font: UIFont) -> [NSAttributedString.Key: Any] {
            [.font: font, .foregroundColor: UIColor.white,
             .strokeColor: UIColor.black, .strokeWidth: -2.5, .shadow: shadow]
        }
        let str = NSMutableAttributedString(string: clip.line1 + "\n", attributes: attrs(f1))
        str.append(NSAttributedString(string: clip.line2 + "\n", attributes: attrs(f2)))
        str.append(NSAttributedString(string: clip.line3, attributes: attrs(f3)))
        layer.string = str
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
