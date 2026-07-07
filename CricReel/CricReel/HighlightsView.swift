//
//  HighlightsView.swift
//  CricReel
//
//  On-demand highlight reels (filter by 4s/6s/wickets) plus the auto over-reels that
//  were rendered during scoring. Reels can be replayed and shared.
//

import SwiftUI
import SwiftData
import AVKit

struct HighlightsView: View {
    @Bindable var match: Match
    @ObservedObject private var builder = HighlightBuilder.shared

    @State private var selectedTags: Set<HighlightTag> = Set(HighlightTag.allCases)
    @State private var reels: [URL] = []
    @State private var playing: PlayableURL?

    var body: some View {
        List {
            Section("Build a Reel") {
                ForEach(HighlightTag.allCases) { tag in
                    Button {
                        toggle(tag)
                    } label: {
                        HStack {
                            Text(tag.displayName).foregroundStyle(.primary)
                            Spacer()
                            Image(systemName: selectedTags.contains(tag) ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(selectedTags.contains(tag) ? .blue : .secondary)
                        }
                    }
                }
                Button {
                    buildReel()
                } label: {
                    if builder.isBuilding {
                        HStack { ProgressView(); Text("Building…") }
                    } else {
                        Label("Build Highlight Reel", systemImage: "film.stack")
                    }
                }
                .disabled(builder.isBuilding || matchingClipFilenames().isEmpty)

                if matchingClipFilenames().isEmpty {
                    Text("No clips match the selected filters yet.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let msg = builder.statusMessage {
                    Text(msg).font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Reels") {
                if reels.isEmpty {
                    Text("No reels yet. Over reels are built automatically; build a custom one above.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(reels, id: \.self) { url in
                    HStack {
                        Button {
                            playing = PlayableURL(url: url)
                        } label: {
                            Label(url.deletingPathExtension().lastPathComponent, systemImage: "play.rectangle")
                                .foregroundStyle(.primary)
                        }
                        Spacer()
                        ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                    }
                }
                .onDelete(perform: deleteReels)
            }
        }
        .navigationTitle("Highlights")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear(perform: loadReels)
        .onChange(of: builder.lastReelURL) { _, _ in loadReels() }
        .sheet(item: $playing) { item in
            ReelPlayerView(url: item.url)
        }
    }

    private func toggle(_ tag: HighlightTag) {
        if selectedTags.contains(tag) { selectedTags.remove(tag) } else { selectedTags.insert(tag) }
    }

    private func matchingClipFilenames() -> [String] {
        let tags = selectedTags.isEmpty ? Set(HighlightTag.allCases) : selectedTags
        return match.innings
            .sorted { $0.order < $1.order }
            .flatMap { $0.orderedDeliveries }
            .filter { !Set($0.highlightTags).isDisjoint(with: tags) }
            .compactMap { $0.clipFilename }
            .filter { ClipStore.clipExists($0) }
    }

    private func buildReel() {
        let filenames = matchingClipFilenames()
        let name = "reel_\(match.teamAName)_\(match.teamBName)"
            .replacingOccurrences(of: " ", with: "-")
        Task { _ = await builder.buildReel(clipFilenames: filenames, name: name); loadReels() }
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

    var body: some View {
        NavigationStack {
            VideoPlayer(player: AVPlayer(url: url))
                .ignoresSafeArea(edges: .bottom)
                .navigationTitle("Reel")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                    ToolbarItem(placement: .topBarLeading) { ShareLink(item: url) }
                }
        }
    }
}
