//
//  ScoringEntryView.swift
//  CricReel
//
//  The ball-entry screen, presented when a delivery is auto-detected or the scorer taps
//  "Add Ball". Detection + recording are paused while this is on screen.
//

import SwiftUI
import AVKit

struct ScoringEntryView: View {
    let pendingClipFilename: String?
    let bowlerName: String
    let strikerID: UUID
    let nonStrikerID: UUID
    let strikerName: String
    let nonStrikerName: String
    let fieldingSideIDs: [UUID]
    let availableBatters: [UUID]
    let isLastBallOfOver: Bool
    let lookup: PlayerLookup

    var onCommit: (BallInput) -> Void
    var onDiscardClip: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var showingWicket = false
    @State private var wicketPreset: DismissalType = .bowled
    @State private var wicketExtra: ExtraType = .none

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if let clip = pendingClipFilename {
                        VStack(spacing: 8) {
                            // Clips are portrait (9:16, capture is rotated 90°). A stable
                            // looping player auto-plays the delivery; height is capped so
                            // the whole scoring UI fits on a compact phone.
                            ClipPreviewPlayer(url: ClipStore.url(forClip: clip))
                                .aspectRatio(9.0 / 16.0, contentMode: .fit)
                                .frame(maxWidth: .infinity, maxHeight: 210)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                            Button(role: .destructive) {
                                onDiscardClip(); dismiss()
                            } label: {
                                Label("Not a delivery — discard clip", systemImage: "trash")
                            }
                            .font(.subheadline)
                        }
                    }

                    HStack {
                        Label("\(bowlerName) to \(strikerName)", systemImage: "figure.cricket")
                            .font(.subheadline).foregroundStyle(.secondary)
                        Spacer()
                    }

                    ScoringPad(
                        onScore: { extra, runs in
                            var input = BallInput()
                            input.extra = extra
                            input.padRuns = runs
                            onCommit(input)
                            dismiss()
                        },
                        onWicket: { extra, type in
                            wicketExtra = extra
                            wicketPreset = type
                            showingWicket = true
                        })
                }
                .padding()
            }
            .navigationTitle("Score Ball")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
            .sheet(isPresented: $showingWicket) {
                WicketSheet(
                    presetDismissal: wicketPreset,
                    extra: wicketExtra,
                    strikerID: strikerID,
                    nonStrikerID: nonStrikerID,
                    strikerName: strikerName,
                    nonStrikerName: nonStrikerName,
                    fieldingSideIDs: fieldingSideIDs,
                    availableBatters: availableBatters,
                    isLastBallOfOver: isLastBallOfOver,
                    lookup: lookup,
                    clipFilename: pendingClipFilename,
                    onCommit: { input in onCommit(input); dismiss() },
                    onCancel: {},
                    onDiscardClip: pendingClipFilename != nil ? { onDiscardClip(); dismiss() } : nil)
            }
        }
    }
}

/// A stable, muted, seamlessly-looping clip player. Created once (not rebuilt on every
/// SwiftUI redraw), so it reliably shows the whole delivery instead of a poster frame.
private struct ClipPreviewPlayer: View {
    let url: URL
    @State private var player = AVQueuePlayer()
    @State private var looper: AVPlayerLooper?

    var body: some View {
        VideoPlayer(player: player)
            .onAppear {
                if looper == nil {
                    looper = AVPlayerLooper(player: player, templateItem: AVPlayerItem(url: url))
                }
                player.isMuted = true
                player.play()
            }
            .onDisappear { player.pause() }
    }
}
