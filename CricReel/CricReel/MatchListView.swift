//
//  MatchListView.swift
//  CricReel
//
//  Home list of matches + per-match hub (score, commentary, scorecard, highlights).
//

import SwiftUI
import SwiftData

struct MatchListView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \Match.date, order: .reverse) private var matches: [Match]

    @State private var showingSetup = false
    @State private var navPath = NavigationPath()

    var body: some View {
        NavigationStack(path: $navPath) {
            List {
                if matches.isEmpty {
                    ContentUnavailableView("No Matches",
                                           systemImage: "sportscourt",
                                           description: Text("Tap + to create a match."))
                }
                ForEach(matches) { match in
                    NavigationLink(value: match) {
                        MatchRow(match: match)
                    }
                }
                .onDelete { offsets in
                    for i in offsets { context.delete(matches[i]) }
                }
            }
            .navigationTitle("Matches")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showingSetup = true } label: { Image(systemName: "plus") }
                }
            }
            .navigationDestination(for: Match.self) { match in
                MatchHomeView(match: match)
            }
            .sheet(isPresented: $showingSetup) {
                MatchSetupView { created in
                    navPath.append(created)
                }
            }
        }
    }
}

private struct MatchRow: View {
    let match: Match
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(match.teamAName) vs \(match.teamBName)").font(.headline)
            HStack(spacing: 8) {
                statusBadge
                Text(match.date, style: .date).font(.caption).foregroundStyle(.secondary)
                if !match.venue.isEmpty {
                    Text("• \(match.venue)").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var statusBadge: some View {
        Text(match.status.rawValue.capitalized)
            .font(.caption2).bold()
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.2))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }
    private var color: Color {
        switch match.status {
        case .setup: return .orange
        case .live: return .green
        case .completed: return .gray
        }
    }
}

struct MatchHomeView: View {
    @Bindable var match: Match

    var body: some View {
        List {
            Section("Score") {
                ForEach(match.innings.sorted { $0.order < $1.order }) { innings in
                    InningsSummaryRow(match: match, innings: innings)
                }
            }

            if let result = matchResult {
                Section {
                    Label(result, systemImage: "trophy")
                        .font(.headline)
                }
            }

            Section {
                if match.status != .completed {
                    NavigationLink {
                        ScoringView(match: match)
                    } label: {
                        Label("Score Live", systemImage: "dot.radiowaves.left.and.right")
                    }
                }
                NavigationLink {
                    CommentaryView(match: match)
                } label: {
                    Label("Ball-by-Ball", systemImage: "text.line.first.and.arrowtriangle.forward")
                }
                NavigationLink {
                    ScorecardView(match: match)
                } label: {
                    Label("Scorecard", systemImage: "tablecells")
                }
                NavigationLink {
                    HighlightsView(match: match)
                } label: {
                    Label("Highlights", systemImage: "film.stack")
                }
            }

            if match.status != .completed {
                Section {
                    Button(role: .destructive) {
                        match.status = .completed
                    } label: {
                        Label("Mark Match Completed", systemImage: "flag.checkered")
                    }
                }
            }
        }
        .navigationTitle("\(match.teamAName) vs \(match.teamBName)")
        .navigationBarTitleDisplayMode(.inline)
    }

    /// Human-readable result once both innings exist and the match is decided.
    private var matchResult: String? {
        guard match.status == .completed || match.innings.contains(where: { $0.order == 2 }) else { return nil }
        guard let first = match.innings.first(where: { $0.order == 1 }),
              let second = match.innings.first(where: { $0.order == 2 }) else { return nil }
        let s1 = MatchScoring.state(for: first, in: match)
        let s2 = MatchScoring.state(for: second, in: match)
        let firstName = first.battingTeamIsA ? match.teamAName : match.teamBName
        let secondName = second.battingTeamIsA ? match.teamAName : match.teamBName
        if s2.totalRuns > s1.totalRuns {
            let wktsLeft = max(0, match.playersPerSide - 1 - s2.wickets)
            return "\(secondName) won by \(wktsLeft) wicket\(wktsLeft == 1 ? "" : "s")"
        }
        if !second.isComplete { return nil }
        if s1.totalRuns > s2.totalRuns {
            let by = s1.totalRuns - s2.totalRuns
            return "\(firstName) won by \(by) run\(by == 1 ? "" : "s")"
        }
        return "Match tied"
    }
}

/// One line of the innings score summary on the match hub.
private struct InningsSummaryRow: View {
    let match: Match
    let innings: Innings

    var body: some View {
        let battingName = innings.battingTeamIsA ? match.teamAName : match.teamBName
        let state = MatchScoring.state(for: innings, in: match)
        return HStack {
            Text(battingName)
            Spacer()
            Text("\(state.totalRuns)/\(state.wickets)")
                .bold().monospacedDigit()
            Text("(\(state.oversDisplay))")
                .foregroundStyle(.secondary).monospacedDigit()
        }
    }
}
