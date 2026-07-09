//
//  HighlightsView.swift
//  CricReel
//
//  Default reels (each batting team's innings) + custom reels (pick 4s/6s/wickets from
//  either or both innings). Reels can be replayed and shared.
//

import SwiftUI
import SwiftData
import AVKit

struct HighlightsView: View {
    @Bindable var match: Match
    @ObservedObject private var builder = HighlightBuilder.shared
    @Query(sort: \Player.name) private var players: [Player]

    @State private var selectedTags: Set<HighlightTag> = Set(HighlightTag.allCases)
    @State private var selectedInnings: Set<Int> = [1, 2]
    @State private var reels: [URL] = []
    @State private var playing: PlayableURL?

    private var lookup: PlayerLookup { PlayerLookup(players) }
    private var sortedInnings: [Innings] { match.innings.sorted { $0.order < $1.order } }
    private var matchLabel: String {
        "\(match.teamAName)-v-\(match.teamBName)".replacingOccurrences(of: " ", with: "")
    }

    var body: some View {
        List {
            defaultSection
            customSection
            reelsSection
        }
        .navigationTitle("Highlights")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear(perform: loadReels)
        .onChange(of: builder.lastReelURL) { _, _ in loadReels() }
        .sheet(item: $playing) { ReelPlayerView(url: $0.url) }
    }

    // MARK: - Default reels

    private var defaultSection: some View {
        Section("Default Reels") {
            ForEach(sortedInnings) { innings in
                let name = innings.battingTeamIsA ? match.teamAName : match.teamBName
                let clips = builder.reelClips(from: highlightDeliveries(innings, tags: Set(HighlightTag.allCases)),
                                              match: match, lookup: lookup)
                Button {
                    buildReel(clips: clips, name: "\(matchLabel)_\(name.replacingOccurrences(of: " ", with: ""))")
                } label: {
                    HStack {
                        Label("\(name) — Innings \(innings.order)", systemImage: "film")
                        Spacer()
                        Text(clips.isEmpty ? "No clips" : "\(clips.count)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .disabled(clips.isEmpty || builder.isBuilding)
            }
        }
    }

    // MARK: - Custom reel

    private var customSection: some View {
        Section("Custom Reel") {
            ForEach(HighlightTag.allCases) { tag in
                toggleRow(tag.displayName, isOn: selectedTags.contains(tag)) { toggleTag(tag) }
            }
            ForEach(sortedInnings) { innings in
                let name = innings.battingTeamIsA ? match.teamAName : match.teamBName
                toggleRow("\(name) innings", isOn: selectedInnings.contains(innings.order)) {
                    toggleInnings(innings.order)
                }
            }
            Button {
                buildReel(clips: customClips, name: "\(matchLabel)_custom")
            } label: {
                if builder.isBuilding {
                    HStack { ProgressView(); Text("Building…") }
                } else {
                    Label("Build Custom Reel", systemImage: "slider.horizontal.3")
                }
            }
            .disabled(customClips.isEmpty || builder.isBuilding)
            if let msg = builder.statusMessage {
                Text(msg).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Reels list

    private var reelsSection: some View {
        Section("Saved Reels") {
            if reels.isEmpty {
                Text("Over reels build automatically; create match reels above.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            ForEach(reels, id: \.self) { url in
                HStack {
                    Button {
                        playing = PlayableURL(url: url)
                    } label: {
                        Label(url.deletingPathExtension().lastPathComponent, systemImage: "play.rectangle")
                            .foregroundStyle(.primary).lineLimit(1)
                    }
                    Spacer()
                    ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                }
            }
            .onDelete(perform: deleteReels)
        }
    }

    // MARK: - Helpers

    private func toggleRow(_ title: String, isOn: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                Text(title).foregroundStyle(.primary)
                Spacer()
                Image(systemName: isOn ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isOn ? .blue : .secondary)
            }
        }
    }

    private func highlightDeliveries(_ innings: Innings, tags: Set<HighlightTag>) -> [Delivery] {
        innings.orderedDeliveries.filter { !Set($0.highlightTags).isDisjoint(with: tags) }
    }

    private var customClips: [ReelClip] {
        let tags = selectedTags.isEmpty ? Set(HighlightTag.allCases) : selectedTags
        let deliveries = sortedInnings
            .filter { selectedInnings.contains($0.order) }
            .flatMap { highlightDeliveries($0, tags: tags) }
        return builder.reelClips(from: deliveries, match: match, lookup: lookup)
    }

    private func toggleTag(_ tag: HighlightTag) {
        if selectedTags.contains(tag) { selectedTags.remove(tag) } else { selectedTags.insert(tag) }
    }
    private func toggleInnings(_ order: Int) {
        if selectedInnings.contains(order) { selectedInnings.remove(order) } else { selectedInnings.insert(order) }
    }

    private func buildReel(clips: [ReelClip], name: String) {
        Task { _ = await builder.buildReel(clips: clips, name: name); loadReels() }
    }

    private func loadReels() {
        let all = (try? FileManager.default.contentsOfDirectory(
            at: ClipStore.highlightsDirectory,
            includingPropertiesForKeys: [.creationDateKey])) ?? []
        reels = all
            .filter { $0.pathExtension.lowercased() == "mp4" }
            .sorted {
                let d1 = (try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                let d2 = (try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                return d1 > d2
            }
    }

    private func deleteReels(_ offsets: IndexSet) {
        for i in offsets { try? FileManager.default.removeItem(at: reels[i]) }
        loadReels()
    }
}

struct PlayableURL: Identifiable {
    let url: URL
    var id: String { url.absoluteString }
}

struct ReelPlayerView: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss
    @StateObject private var controller = LoopingPlayer()

    private let speeds: [Float] = [1.0, 0.5, 0.25]

    var body: some View {
        NavigationStack {
            VideoPlayer(player: controller.player)
                .ignoresSafeArea(edges: .bottom)
                .navigationTitle("Reel")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) { Button("Done") { dismiss() } }
                    ToolbarItem(placement: .topBarTrailing) {
                        Menu {
                            ForEach(speeds, id: \.self) { sp in
                                Button { controller.setRate(sp) } label: {
                                    Label(speedLabel(sp), systemImage: controller.rate == sp ? "checkmark" : "")
                                }
                            }
                        } label: {
                            Label(speedLabel(controller.rate), systemImage: "speedometer")
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) { ShareLink(item: url) }
                }
                .onAppear { controller.start(url: url) }
                .onDisappear { controller.stop() }
        }
    }

    private func speedLabel(_ sp: Float) -> String {
        sp == 1.0 ? "1×" : (sp == 0.5 ? "0.5×" : "0.25×")
    }
}
