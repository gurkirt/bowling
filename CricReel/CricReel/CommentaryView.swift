//
//  CommentaryView.swift
//  CricReel
//
//  Ball-by-ball commentary with colour-coded outcomes (4 green, 6 purple, W red).
//  Tapping a ball with a clip replays it with a delivery overlay.
//

import SwiftUI
import SwiftData
import AVKit

struct CommentaryView: View {
    @Bindable var match: Match
    @Query(sort: \Player.name) private var players: [Player]

    @State private var playing: ClipItem?

    private var lookup: PlayerLookup { PlayerLookup(players) }

    var body: some View {
        List {
            ForEach(match.innings.sorted { $0.order > $1.order }) { innings in
                Section(header: Text(inningsTitle(innings))) {
                    let balls = Array(innings.orderedDeliveries.reversed())
                    if balls.isEmpty {
                        Text("No balls yet.").foregroundStyle(.secondary)
                    }
                    ForEach(balls) { ball in
                        row(ball)
                    }
                }
            }
        }
        .navigationTitle("Ball-by-Ball")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $playing) { ClipPlayerView(item: $0) }
    }

    private func row(_ ball: Delivery) -> some View {
        let kind = DeliveryFormatting.kind(ball)
        let hasClip = ClipStore.clipExists(ball.clipFilename)
        return HStack(spacing: 12) {
            Text(DeliveryFormatting.badge(ball))
                .font(.caption.bold()).monospacedDigit()
                .frame(width: 30, height: 30)
                .background(kind.color, in: Circle())
                .foregroundStyle(.white)
            VStack(alignment: .leading, spacing: 2) {
                Text("\(ball.overNumber).\(ball.ballInOver)")
                    .font(.caption2).foregroundStyle(.secondary).monospacedDigit()
                Text(DeliveryFormatting.description(ball, lookup: lookup))
                    .font(.subheadline)
            }
            Spacer()
            if hasClip {
                Button {
                    let (l1, l2) = DeliveryFormatting.overlayLines(ball, lookup: lookup)
                    playing = ClipItem(filename: ball.clipFilename!, line1: l1, line2: l2)
                } label: {
                    Image(systemName: "play.circle.fill").font(.title2)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.vertical, 2)
    }

    private func inningsTitle(_ innings: Innings) -> String {
        let name = innings.battingTeamIsA ? match.teamAName : match.teamBName
        return "Innings \(innings.order) — \(name)"
    }
}

struct ClipItem: Identifiable {
    let filename: String
    let line1: String
    let line2: String
    var id: String { filename }
}

struct ClipPlayerView: View {
    let item: ClipItem
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            GeometryReader { geo in
                VideoPlayer(player: AVPlayer(url: ClipStore.url(forClip: item.filename)))
                    .overlay(alignment: .top) {
                        VStack(spacing: 4) {
                            Text(item.line1).font(.headline)
                            Text(item.line2).font(.subheadline)
                        }
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.white)
                        .shadow(radius: 6)
                        .padding(.horizontal, 20)
                        .padding(.top, geo.size.height * 0.12)
                    }
            }
            .ignoresSafeArea(edges: .bottom)
            .navigationTitle("Replay")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}
