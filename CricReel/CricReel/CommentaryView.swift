//
//  CommentaryView.swift
//  CricReel
//
//  Ball-by-ball commentary grouped by over, with an end-of-over summary (score, over
//  runs/wickets, CRR and — chasing — required runs & RRR). Colour-coded outcomes.
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
                    let groups = overGroups(innings)
                    if groups.isEmpty {
                        Text("No balls yet.").foregroundStyle(.secondary)
                    }
                    ForEach(groups.reversed()) { group in
                        overSummary(group, innings: innings)
                        ForEach(group.balls.reversed()) { ball in row(ball) }
                    }
                }
            }
        }
        .navigationTitle("Ball-by-Ball")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $playing) { ClipPlayerView(item: $0) }
    }

    // MARK: - Over summary

    private func overSummary(_ group: OverGroup, innings: Innings) -> some View {
        let target = MatchScoring.target(for: innings, in: match)
        let totalBalls = match.oversPerInnings * match.ballsPerOver
        let ballsLeft = max(0, totalBalls - group.cumLegalBalls)
        let crr = group.cumLegalBalls == 0 ? 0 : Double(group.cumRuns) / (Double(group.cumLegalBalls) / 6.0)
        return VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline) {
                Text("End of Over \(group.overNumber + 1)")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("\(group.cumRuns)/\(group.cumWkts)")
                    .font(.title3.bold()).monospacedDigit()
            }
            HStack {
                Text("Over: \(group.runsInOver) run\(group.runsInOver == 1 ? "" : "s")"
                     + (group.wktsInOver > 0 ? ", \(group.wktsInOver) wkt" : ""))
                Spacer()
                if let t = target, ballsLeft > 0 {
                    let need = max(0, t - group.cumRuns)
                    let rrr = Double(need) / (Double(ballsLeft) / 6.0)
                    Text("Need \(need) off \(ballsLeft) · RRR \(fmt(rrr))")
                } else {
                    Text("CRR \(fmt(crr))")
                }
            }
            .font(.caption).foregroundStyle(.secondary).monospacedDigit()
        }
        .padding(.vertical, 4)
        .listRowBackground(Color(.secondarySystemGroupedBackground).opacity(0.6))
    }

    // MARK: - Ball row

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
                    playing = ClipItem(filename: ball.clipFilename!,
                                       info: ClipOverlay.info(for: ball, match: match, lookup: lookup))
                } label: {
                    Image(systemName: "play.circle.fill").font(.title2)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: - Grouping

    struct OverGroup: Identifiable {
        let overNumber: Int
        let balls: [Delivery]
        let runsInOver: Int
        let wktsInOver: Int
        let cumRuns: Int
        let cumWkts: Int
        let cumLegalBalls: Int
        var id: Int { overNumber }
    }

    private func overGroups(_ innings: Innings) -> [OverGroup] {
        let grouped = Dictionary(grouping: innings.orderedDeliveries, by: { $0.overNumber })
        var cumRuns = 0, cumWkts = 0, cumBalls = 0
        var result: [OverGroup] = []
        for overNo in grouped.keys.sorted() {
            let balls = grouped[overNo]!.sorted { $0.sequence < $1.sequence }
            let runsInOver = balls.reduce(0) { $0 + $1.totalRuns }
            let wktsInOver = balls.filter { $0.isWicket }.count
            cumRuns += runsInOver
            cumWkts += wktsInOver
            cumBalls += balls.filter { $0.isLegalDelivery }.count
            result.append(OverGroup(overNumber: overNo, balls: balls,
                                    runsInOver: runsInOver, wktsInOver: wktsInOver,
                                    cumRuns: cumRuns, cumWkts: cumWkts, cumLegalBalls: cumBalls))
        }
        return result
    }

    private func inningsTitle(_ innings: Innings) -> String {
        let name = innings.battingTeamIsA ? match.teamAName : match.teamBName
        return "Innings \(innings.order) — \(name)"
    }
    private func fmt(_ v: Double) -> String { String(format: "%.2f", v) }
}

struct ClipItem: Identifiable {
    let filename: String
    let info: OverlayInfo
    var id: String { filename }
}

struct ClipPlayerView: View {
    let item: ClipItem
    @Environment(\.dismiss) private var dismiss
    @State private var isPreparingShare = false
    @State private var shareURL: PlayableURL?

    var body: some View {
        NavigationStack {
            GeometryReader { geo in
                VideoPlayer(player: AVPlayer(url: ClipStore.url(forClip: item.filename)))
                    .overlay(alignment: .top) {
                        ClipOverlayView(info: item.info)
                            .padding(.horizontal, 16)
                            .padding(.top, geo.size.height * 0.10)
                    }
            }
            .ignoresSafeArea(edges: .bottom)
            .navigationTitle("Replay")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { Task { await prepareShare() } } label: {
                        if isPreparingShare { ProgressView() }
                        else { Image(systemName: "square.and.arrow.up") }
                    }
                    .disabled(isPreparingShare)
                }
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .sheet(item: $shareURL) { ShareSheet(url: $0.url) }
        }
    }

    private func prepareShare() async {
        isPreparingShare = true
        let clip = ReelClip(url: ClipStore.url(forClip: item.filename), info: item.info)
        if let url = await HighlightBuilder.shared.exportCaptionedClip(clip) {
            shareURL = PlayableURL(url: url)
        }
        isPreparingShare = false
    }
}
