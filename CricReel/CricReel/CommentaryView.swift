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
import AVFoundation

struct CommentaryView: View {
    @Bindable var match: Match
    @Query(sort: \Player.name) private var players: [Player]

    @State private var playing: ClipItem?
    /// Empty = all balls; otherwise only deliveries carrying one of these tags
    /// (fours / sixes / wickets) are listed — and swiped through in the player.
    @State private var filterTags: Set<HighlightTag> = []

    private var lookup: PlayerLookup { PlayerLookup(players) }

    private func matchesFilter(_ ball: Delivery) -> Bool {
        filterTags.isEmpty || !Set(ball.highlightTags).isDisjoint(with: filterTags)
    }

    var body: some View {
        List {
            ForEach(match.innings.sorted { $0.order > $1.order }) { innings in
                Section(header: Text(inningsTitle(innings))) {
                    let groups = overGroups(innings)
                    if groups.isEmpty {
                        Text("No balls yet.").foregroundStyle(.secondary)
                    } else if !groups.contains(where: { $0.balls.contains(where: matchesFilter) }) {
                        Text("No matching deliveries.").foregroundStyle(.secondary)
                    }
                    ForEach(groups.reversed()) { group in
                        let balls = group.balls.filter(matchesFilter)
                        if !balls.isEmpty {
                            // Over summaries only make sense for the unfiltered feed.
                            if filterTags.isEmpty { overSummary(group, innings: innings) }
                            ForEach(balls.reversed()) { ball in row(ball) }
                        }
                    }
                }
            }
        }
        .navigationTitle("Ball-by-Ball")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button { filterTags = [] } label: {
                        if filterTags.isEmpty { Label("All Balls", systemImage: "checkmark") }
                        else { Text("All Balls") }
                    }
                    Divider()
                    ForEach(HighlightTag.allCases) { tag in
                        Button { toggleFilter(tag) } label: {
                            if filterTags.contains(tag) { Label(tag.displayName, systemImage: "checkmark") }
                            else { Text(tag.displayName) }
                        }
                    }
                } label: {
                    Image(systemName: filterTags.isEmpty
                          ? "line.3.horizontal.decrease.circle"
                          : "line.3.horizontal.decrease.circle.fill")
                }
            }
        }
        // Full-screen cover, not a sheet: a sheet's drag-to-dismiss would swallow
        // the vertical swipes used to move between clips.
        .fullScreenCover(item: $playing) { item in
            let playlist = clipPlaylist
            ClipPlayerView(items: playlist,
                           startIndex: playlist.firstIndex(where: { $0.id == item.id }) ?? 0)
        }
    }

    /// Clip-bearing deliveries in match order, honouring the active filter —
    /// the swipe-through playlist.
    private var clipPlaylist: [ClipItem] {
        match.innings.sorted { $0.order < $1.order }.flatMap { innings in
            innings.orderedDeliveries.compactMap { ball -> ClipItem? in
                guard matchesFilter(ball),
                      let f = ball.clipFilename, ClipStore.clipExists(f) else { return nil }
                return ClipItem(filename: f,
                                info: ClipOverlay.info(for: ball, match: match, lookup: lookup))
            }
        }
    }

    private func toggleFilter(_ tag: HighlightTag) {
        if filterTags.contains(tag) { filterTags.remove(tag) } else { filterTags.insert(tag) }
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

/// AVPlayer wrapper that autoplays, loops, and supports slow-motion playback.
final class LoopingPlayer: ObservableObject {
    let player = AVPlayer()
    @Published private(set) var rate: Float = 1.0
    /// Increments every time playback wraps — lets overlays re-run their reveal.
    @Published private(set) var loopCount = 0
    private var endObserver: NSObjectProtocol?

    func start(url: URL) {
        // Called again when swiping between clips — drop the previous item's observer.
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        let playerItem = AVPlayerItem(url: url)
        player.replaceCurrentItem(with: playerItem)
        player.actionAtItemEnd = .none
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: playerItem, queue: .main) { [weak self] _ in
            guard let self else { return }
            self.player.seek(to: .zero)
            self.player.playImmediately(atRate: self.rate)
            self.loopCount += 1
        }
        player.playImmediately(atRate: rate)
    }

    func setRate(_ r: Float) {
        rate = r
        player.playImmediately(atRate: r)
    }

    func stop() {
        player.pause()
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        endObserver = nil
    }
}

struct ClipPlayerView: View {
    let items: [ClipItem]
    @State private var index: Int
    @Environment(\.dismiss) private var dismiss
    @StateObject private var controller = LoopingPlayer()
    @State private var isPreparingShare = false
    @State private var shareURL: PlayableURL?

    private let speeds: [Float] = [1.0, 0.5, 0.25]

    init(items: [ClipItem], startIndex: Int) {
        self.items = items
        _index = State(initialValue: min(max(0, startIndex), max(0, items.count - 1)))
    }

    private var item: ClipItem { items[index] }

    var body: some View {
        NavigationStack {
            VideoPlayer(player: controller.player)
                .overlay {
                    // .id ties the overlay to both the clip and the loop pass, so the
                    // reveal sequence (before-score → result) replays after every swipe
                    // and every loop. The delay stretches with slow-motion.
                    ClipOverlayView(info: item.info,
                                    revealDelay: SettingsStore.shared.overlayRevealDelay / Double(controller.rate))
                        .id("\(index)-\(controller.loopCount)")
                }
                .overlay(alignment: .bottom) { swipeHint }
                .ignoresSafeArea(edges: .bottom)
                // Swipe up → next ball, swipe down → previous ball. Simultaneous so
                // AVKit's own controls don't block it.
                .simultaneousGesture(
                    DragGesture(minimumDistance: 30)
                        .onEnded { v in
                            guard abs(v.translation.height) > abs(v.translation.width),
                                  abs(v.translation.height) > 60 else { return }
                            step(v.translation.height < 0 ? 1 : -1)
                        }
                )
            .navigationTitle("Replay · \(index + 1)/\(items.count)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        ForEach(speeds, id: \.self) { sp in
                            Button { controller.setRate(sp) } label: {
                                if controller.rate == sp {
                                    Label(speedLabel(sp), systemImage: "checkmark")
                                } else {
                                    Text(speedLabel(sp))
                                }
                            }
                        }
                    } label: {
                        Label(speedLabel(controller.rate), systemImage: "speedometer")
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { Task { await prepareShare() } } label: {
                        if isPreparingShare { ProgressView() }
                        else { Image(systemName: "square.and.arrow.up") }
                    }
                    .disabled(isPreparingShare)
                }
            }
            .sheet(item: $shareURL) { ShareSheet(url: $0.url) }
            .onAppear { controller.start(url: ClipStore.url(forClip: item.filename)) }
            .onDisappear { controller.stop() }
        }
    }

    /// Move through the playlist; loads the neighbouring clip and replays its reveal.
    private func step(_ delta: Int) {
        let next = index + delta
        guard items.indices.contains(next) else { return }
        withAnimation(.easeInOut(duration: 0.15)) { index = next }
        controller.start(url: ClipStore.url(forClip: items[next].filename))
    }

    /// Navigation pill: tappable chevrons (previous/next ball) around the position.
    @ViewBuilder
    private var swipeHint: some View {
        if items.count > 1 {
            HStack(spacing: 2) {
                Button { step(-1) } label: {
                    Image(systemName: "chevron.down")
                        .frame(width: 40, height: 34)
                }
                .disabled(index == 0)
                .opacity(index == 0 ? 0.3 : 1)

                Text("\(index + 1) / \(items.count)")
                    .monospacedDigit()
                    .font(.footnote.weight(.semibold))

                Button { step(1) } label: {
                    Image(systemName: "chevron.up")
                        .frame(width: 40, height: 34)
                }
                .disabled(index + 1 >= items.count)
                .opacity(index + 1 >= items.count ? 0.3 : 1)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 6)
            .background(.black.opacity(0.45), in: Capsule())
            .padding(.bottom, 90)
        }
    }

    private func speedLabel(_ sp: Float) -> String {
        sp == 1.0 ? "1×" : (sp == 0.5 ? "0.5×" : "0.25×")
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
