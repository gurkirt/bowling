//
//  HighlightBuilder.swift
//  CricReel
//
//  Stitches per-ball clips into highlight reels using AVFoundation. All rendering runs
//  off the scoring path (async) so live scoring is never blocked.
//

import Foundation
import AVFoundation
import Combine

@MainActor
final class HighlightBuilder: ObservableObject {
    static let shared = HighlightBuilder()

    @Published var isBuilding = false
    @Published var statusMessage: String?
    @Published var lastReelURL: URL?

    private init() {}

    /// Auto reel for a completed over: only its highlight deliveries (4s/6s/wickets).
    func buildOverReel(match: Match, innings: Innings, overNumber: Int) {
        let urls = innings.orderedDeliveries
            .filter { $0.overNumber == overNumber && !$0.highlightTags.isEmpty }
            .compactMap { $0.clipFilename }
            .filter { ClipStore.clipExists($0) }
            .map { ClipStore.url(forClip: $0) }
        guard !urls.isEmpty else { return }

        let out = ClipStore.highlightsDirectory
            .appendingPathComponent("over_\(innings.order)_\(overNumber + 1)_\(shortID()).mp4")
        render(urls: urls, output: out, doneMessage: "Over \(overNumber + 1) reel ready")
    }

    /// On-demand reel from an explicit list of clip filenames (gathered by the caller).
    /// Returns the output URL, or nil on failure.
    func buildReel(clipFilenames: [String], name: String) async -> URL? {
        let urls = clipFilenames
            .filter { ClipStore.clipExists($0) }
            .map { ClipStore.url(forClip: $0) }
        guard !urls.isEmpty else { return nil }

        let out = ClipStore.highlightsDirectory
            .appendingPathComponent("\(name)_\(shortID()).mp4")
        isBuilding = true
        statusMessage = "Building reel…"
        do {
            let result = try await Self.stitch(clipURLs: urls, outputURL: out)
            isBuilding = false
            statusMessage = "Reel ready"
            lastReelURL = result
            return result
        } catch {
            isBuilding = false
            statusMessage = "Reel failed: \(error.localizedDescription)"
            return nil
        }
    }

    // MARK: - Internal

    private func render(urls: [URL], output: URL, doneMessage: String) {
        isBuilding = true
        statusMessage = "Building reel…"
        Task {
            do {
                let result = try await Self.stitch(clipURLs: urls, outputURL: output)
                self.lastReelURL = result
                self.statusMessage = doneMessage
            } catch {
                self.statusMessage = "Reel failed: \(error.localizedDescription)"
            }
            self.isBuilding = false
        }
    }

    private func shortID() -> String { String(UUID().uuidString.prefix(6)) }

    /// Concatenate the given (silent) clips into a single MP4.
    nonisolated static func stitch(clipURLs: [URL], outputURL: URL) async throws -> URL {
        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(
            withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw ReelError.trackCreationFailed
        }

        var cursor = CMTime.zero
        var appliedTransform = false

        for url in clipURLs {
            let asset = AVURLAsset(url: url)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let src = tracks.first else { continue }
            let duration = try await asset.load(.duration)
            let range = CMTimeRange(start: .zero, duration: duration)
            try videoTrack.insertTimeRange(range, of: src, at: cursor)
            if !appliedTransform {
                videoTrack.preferredTransform = try await src.load(.preferredTransform)
                appliedTransform = true
            }
            cursor = CMTimeAdd(cursor, duration)
        }

        guard cursor > .zero else { throw ReelError.noUsableClips }

        try? FileManager.default.removeItem(at: outputURL)
        guard let export = AVAssetExportSession(
            asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw ReelError.exportSetupFailed
        }
        export.outputURL = outputURL
        export.outputFileType = .mp4
        export.shouldOptimizeForNetworkUse = true

        return try await withCheckedThrowingContinuation { continuation in
            export.exportAsynchronously {
                switch export.status {
                case .completed:
                    continuation.resume(returning: outputURL)
                case .failed, .cancelled:
                    continuation.resume(throwing: export.error ?? ReelError.exportFailed)
                default:
                    continuation.resume(throwing: ReelError.exportFailed)
                }
            }
        }
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
