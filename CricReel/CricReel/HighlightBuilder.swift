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

    /// Innings highlights reel, rebuilt at each over end from every tagged delivery so
    /// far. Always writes the same file — "<TEAM>_innings.mp4" — so it grows in place.
    func buildInningsReel(match: Match, innings: Innings, lookup: PlayerLookup) {
        let clips = reelClips(from: innings.orderedDeliveries.filter { !$0.highlightTags.isEmpty },
                              match: match, lookup: lookup)
        guard !clips.isEmpty else { return }
        let team = match.teamReelName(isTeamA: innings.battingTeamIsA)
            .replacingOccurrences(of: " ", with: "")
        let out = ClipStore.highlightsDirectory.appendingPathComponent("\(team)_innings.mp4")
        render(clips: clips, output: out, doneMessage: "\(team) innings reel updated")
    }

    /// Build a reel from explicit clips (gathered by the caller). Returns the URL, or nil.
    /// The name is used as-is ("<name>.mp4") and overwrites any previous build.
    func buildReel(clips: [ReelClip], name: String) async -> URL? {
        guard !clips.isEmpty else { return nil }
        let out = ClipStore.highlightsDirectory.appendingPathComponent("\(name).mp4")
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
        let revealDelay = SettingsStore.shared.overlayRevealDelay
        for seg in segments {
            let ss = CMTimeGetSeconds(seg.start)
            let ee = ss + CMTimeGetSeconds(seg.duration)
            let overlay = makeOverlay(seg.clip.info, renderSize: renderSize,
                                      segStart: ss, segEnd: ee, total: totalSeconds,
                                      revealDelay: revealDelay)
            parentLayer.addSublayer(overlay)
        }
        return (parentLayer, videoLayer)
    }

    /// Full-frame overlay for one clip segment, mirroring the in-app design:
    /// white score chips along the top ([accent bar][team][score][overs] … [opponent]
    /// [score]), "Bowler → Batter" centred until the reveal, then the result in big
    /// white letters mid-frame. Score AND overs chips flip together at the reveal,
    /// `revealDelay` seconds in, capped at 1s before the segment ends.
    @MainActor
    private static func makeOverlay(_ info: OverlayInfo, renderSize: CGSize,
                                    segStart: Double, segEnd: Double, total: Double,
                                    revealDelay: Double) -> CALayer {
        let W = renderSize.width, H = renderSize.height

        let container = CALayer()
        container.frame = CGRect(origin: .zero, size: renderSize)
        container.opacity = 0

        // ── Score chips: two rows each side, inset toward the middle ─────────
        // Row 1: team name (bigger — max six letters). Row 2: score + overs boxes.
        // Sized to match the in-app clip overlay proportions.
        let teamChipH = H * 0.048
        let chipH = H * 0.040
        let teamFont = UIFont.systemFont(ofSize: H * 0.027, weight: .bold)
        let chipFont = UIFont.monospacedDigitSystemFont(ofSize: H * 0.022, weight: .semibold)
        let margin = W * 0.07
        let gap = W * 0.012
        let rowGap = H * 0.008
        let row1Y = H - H * 0.045 - teamChipH   // CA origin is bottom-left
        let row2Y = row1Y - rowGap - chipH

        // Left: batting team above, score + overs (both flip at reveal) beneath.
        let teamChip = chip(info.battingTeam, font: teamFont, height: teamChipH)
        teamChip.frame.origin = CGPoint(x: margin, y: row1Y)
        container.addSublayer(teamChip)

        let scorePair = flipChipPair(before: info.scoreBefore, after: info.scoreAfter,
                                     font: chipFont, height: chipH, at: CGPoint(x: margin, y: row2Y))
        container.addSublayer(scorePair.before)
        container.addSublayer(scorePair.after)

        let oversX = margin + scorePair.before.frame.width + gap
        let oversPair = flipChipPair(before: info.oversBefore, after: info.oversAfter,
                                     font: chipFont, height: chipH, at: CGPoint(x: oversX, y: row2Y))
        container.addSublayer(oversPair.before)
        container.addSublayer(oversPair.after)

        // Right: opponent name above, their score beneath (in the chase) — mirrored.
        let oppChip = chip(info.opponent, font: teamFont, height: teamChipH)
        oppChip.frame.origin = CGPoint(x: W - margin - oppChip.frame.width, y: row1Y)
        container.addSublayer(oppChip)
        if let s = info.opponentScore {
            let c = chip(s, font: chipFont, height: chipH)
            c.frame.origin = CGPoint(x: W - margin - c.frame.width, y: row2Y)
            container.addSublayer(c)
        }

        // ── Centre: delivery until the reveal, then the big result ──────────
        // Delivery / result sit 22% of the way down the frame (CA y is from the
        // bottom, so their centre is at 0.78 · H), clear of the action.
        let textCenterY = H * 0.78
        let deliveryFont = UIFont.systemFont(ofSize: H * 0.029, weight: .semibold)
        let delivery = glowText(info.delivery, font: deliveryFont)
        delivery.frame = CGRect(x: W * 0.05, y: textCenterY - deliveryFont.lineHeight / 2,
                                width: W * 0.9, height: deliveryFont.lineHeight * 1.2)
        container.addSublayer(delivery)

        // The reveal carries the result colour in its text (white for plain runs).
        let resultColor: UIColor = info.accent == .neutral ? .white : info.accent.uiColor
        let resultSize = info.result.count > 8 ? H * 0.052 : H * 0.075
        let resultFont = UIFont.systemFont(ofSize: resultSize, weight: .heavy)
        let result = glowText(info.result, font: resultFont, color: resultColor)
        result.frame = CGRect(x: W * 0.05, y: textCenterY - resultFont.lineHeight / 2,
                              width: W * 0.9, height: resultFont.lineHeight * 1.2)
        result.opacity = 0
        container.addSublayer(result)

        // ── Timing ───────────────────────────────────────────────────────────
        let segDur = segEnd - segStart
        let revealAbs = segStart + min(max(0.2, revealDelay), max(0.2, segDur - 1.0))
        func n(_ t: Double) -> Double { max(0, min(1, t / total)) }
        let s = n(segStart), e = n(segEnd)
        let r = n(revealAbs)
        let rRamp = n(min(revealAbs + 0.25, segEnd))

        // Whole overlay visible only during its segment (hard cut — reliable here).
        container.add(opacityKeyframe([0, 0, 1, 1, 0, 0], [0, s, s, e, e, 1], total, .discrete), forKey: "vis")

        // Before → after flips at the reveal: score, overs, delivery→result.
        let preKeys: [Double] = [0, s, s, r, r, 1]
        let postKeys: [Double] = [0, r, r, e, e, 1]
        let vals: [Double] = [0, 0, 1, 1, 0, 0]
        scorePair.before.add(opacityKeyframe(vals, preKeys, total, .discrete), forKey: "f")
        oversPair.before.add(opacityKeyframe(vals, preKeys, total, .discrete), forKey: "f")
        delivery.add(opacityKeyframe(vals, preKeys, total, .discrete), forKey: "f")
        scorePair.after.add(opacityKeyframe(vals, postKeys, total, .discrete), forKey: "f")
        oversPair.after.add(opacityKeyframe(vals, postKeys, total, .discrete), forKey: "f")
        // The result pops with a quick fade rather than a hard cut.
        result.add(opacityKeyframe(vals, [0, r, rRamp, e, e, 1], total, .linear), forKey: "f")

        return container
    }

    /// A white rounded chip with centred black text. Sharp at export resolution.
    @MainActor
    private static func chip(_ text: String, font: UIFont, height: CGFloat,
                             minWidth: CGFloat = 0) -> CALayer {
        let padH = height * 0.38
        let textW = ceil((text as NSString).size(withAttributes: [.font: font]).width)
        let w = max(textW + padH * 2, minWidth)
        let layer = CALayer()
        layer.frame = CGRect(x: 0, y: 0, width: w, height: height)
        layer.backgroundColor = UIColor.white.cgColor
        layer.cornerRadius = height * 0.2
        let t = CATextLayer()
        t.contentsScale = 3
        t.alignmentMode = .center
        t.isWrapped = false
        t.string = NSAttributedString(string: text, attributes: [
            .font: font, .foregroundColor: UIColor.black])
        t.frame = CGRect(x: 0, y: (height - font.lineHeight) / 2, width: w, height: font.lineHeight)
        layer.addSublayer(t)
        return layer
    }

    /// Two same-width chips stacked at one position; caller animates the flip.
    @MainActor
    private static func flipChipPair(before: String, after: String, font: UIFont,
                                     height: CGFloat, at origin: CGPoint)
        -> (before: CALayer, after: CALayer) {
        let wB = ceil((before as NSString).size(withAttributes: [.font: font]).width)
        let wA = ceil((after as NSString).size(withAttributes: [.font: font]).width)
        let w = max(wB, wA) + height * 0.38 * 2
        let b = chip(before, font: font, height: height, minWidth: w)
        let a = chip(after, font: font, height: height, minWidth: w)
        b.frame.origin = origin
        a.frame.origin = origin
        a.opacity = 0
        return (b, a)
    }

    /// Bright text with a soft drop shadow for legibility over video. No stroke —
    /// strokes muddy the glyphs at export resolution.
    @MainActor
    private static func glowText(_ string: String, font: UIFont,
                                 color: UIColor = .white) -> CATextLayer {
        let layer = CATextLayer()
        layer.contentsScale = 3
        layer.alignmentMode = .center
        layer.isWrapped = false
        layer.truncationMode = .end
        let shadow = NSShadow()
        shadow.shadowColor = UIColor.black.withAlphaComponent(0.9)
        shadow.shadowBlurRadius = 14
        shadow.shadowOffset = CGSize(width: 0, height: 3)
        layer.string = NSAttributedString(string: string, attributes: [
            .font: font,
            .foregroundColor: color,
            .shadow: shadow])
        return layer
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
