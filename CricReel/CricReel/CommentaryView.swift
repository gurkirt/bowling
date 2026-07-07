//
//  CommentaryView.swift
//  CricReel
//
//  Ball-by-ball commentary across all innings; tap a ball with a clip to replay it.
//

import SwiftUI
import SwiftData
import AVKit

struct CommentaryView: View {
    @Bindable var match: Match
    @Query(sort: \Player.name) private var players: [Player]

    @State private var playingClip: ClipItem?

    private var lookup: PlayerLookup { PlayerLookup(players) }

    var body: some View {
        List {
            ForEach(match.innings.sorted { $0.order > $1.order }) { innings in
                Section(header: Text(inningsTitle(innings))) {
                    let balls = innings.orderedDeliveries.reversed()
                    if balls.isEmpty {
                        Text("No balls yet.").foregroundStyle(.secondary)
                    }
                    ForEach(Array(balls)) { ball in
                        CommentaryRow(ball: ball) {
                            if let clip = ball.clipFilename, ClipStore.clipExists(clip) {
                                playingClip = ClipItem(filename: clip)
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Ball-by-Ball")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $playingClip) { item in
            ClipPlayerView(filename: item.filename)
        }
    }

    private func inningsTitle(_ innings: Innings) -> String {
        let name = innings.battingTeamIsA ? match.teamAName : match.teamBName
        return "Innings \(innings.order) — \(name)"
    }
}

private struct CommentaryRow: View {
    let ball: Delivery
    var onPlay: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Text("\(ball.overNumber).\(ball.ballInOver)")
                .font(.caption).monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 34, alignment: .leading)
            VStack(alignment: .leading, spacing: 4) {
                Text(ball.commentary.isEmpty ? "\(ball.totalRuns) run(s)" : ball.commentary)
                HStack(spacing: 6) {
                    ForEach(ball.highlightTags) { tag in
                        Text(tag.displayName)
                            .font(.caption2).bold()
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.blue.opacity(0.15))
                            .clipShape(Capsule())
                    }
                }
            }
            Spacer()
            if let clip = ball.clipFilename, ClipStore.clipExists(clip) {
                Button(action: onPlay) {
                    Image(systemName: "play.circle.fill").font(.title2)
                }
                .buttonStyle(.plain)
            } else {
                Image(systemName: "film.slash")
                    .foregroundStyle(.tertiary)
                    .help("No clip")
            }
        }
        .padding(.vertical, 2)
    }
}

struct ClipItem: Identifiable {
    let filename: String
    var id: String { filename }
}

struct ClipPlayerView: View {
    let filename: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VideoPlayer(player: AVPlayer(url: ClipStore.url(forClip: filename)))
                .ignoresSafeArea(edges: .bottom)
                .navigationTitle("Replay")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
    }
}
