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
    @Query(sort: \Player.name) private var players: [Player]

    private var lookup: PlayerLookup { PlayerLookup(players) }

    var body: some View {
        List {
            Section { summaryCard }

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

    // MARK: - Summary card

    @ViewBuilder private var summaryCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(formatLine).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(match.date, style: .date).font(.caption).foregroundStyle(.secondary)
            }
            if !match.venue.isEmpty {
                Text(match.venue).font(.caption).foregroundStyle(.secondary)
            }
            teamScoreRow(isTeamA: true)
            teamScoreRow(isTeamA: false)
            Divider()
            Text(statusLine).font(.subheadline).fontWeight(.semibold)
            if let info = infoLine {
                Text(info).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            if let cur = liveInnings { atCreaseView(cur) }
        }
        .padding(.vertical, 4)
    }

    private var formatLine: String {
        "\(match.oversPerInnings) overs · \(match.playersPerSide)-a-side"
    }

    private func teamScoreRow(isTeamA: Bool) -> some View {
        let s = MatchScoring.teamScore(isTeamA: isTeamA, in: match)
        let batting = currentInnings?.battingTeamIsA == isTeamA && match.status != .completed
        return HStack {
            HStack(spacing: 6) {
                Circle().fill(batting ? Color.green : Color.clear).frame(width: 7, height: 7)
                Text(match.teamName(isTeamA: isTeamA)).font(.headline)
            }
            Spacer()
            if s.batted {
                Text("\(s.runs)/\(s.wickets)").font(.headline).monospacedDigit()
                Text("(\(s.overs))").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            } else {
                Text("yet to bat").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func atCreaseView(_ innings: Innings) -> some View {
        let s = MatchScoring.state(for: innings, in: match)
        let batting = StatsBuilder.batting(from: innings.orderedDeliveries)
        let bowling = StatsBuilder.bowling(from: innings.orderedDeliveries)
        let bowlerID = innings.orderedDeliveries.last?.bowlerID
        return VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 14) {
                if let st = s.strikerID {
                    Text("\(lookup.name(st))* \(batText(batting[st]))").fontWeight(.medium)
                }
                if let ns = s.nonStrikerID {
                    Text("\(lookup.name(ns)) \(batText(batting[ns]))").foregroundStyle(.secondary)
                }
            }
            if let b = bowlerID {
                Text("Bowler: \(lookup.name(b)) \(bowlText(bowling[b]))").foregroundStyle(.secondary)
            }
        }
        .font(.caption).monospacedDigit()
    }

    private func batText(_ l: BattingLine?) -> String { "\(l?.runs ?? 0)(\(l?.ballsFaced ?? 0))" }
    private func bowlText(_ l: BowlingLine?) -> String {
        "\(l?.wickets ?? 0)-\(l?.runsConceded ?? 0) (\(l?.oversDisplay ?? "0.0"))"
    }

    // MARK: - Derived

    private var currentInnings: Innings? {
        let sorted = match.innings.sorted { $0.order < $1.order }
        return sorted.first(where: { !$0.isComplete }) ?? sorted.last
    }

    private var liveInnings: Innings? {
        guard match.status != .completed, let cur = currentInnings, !cur.deliveries.isEmpty else { return nil }
        return cur
    }

    private var statusLine: String {
        if let r = matchResult { return r }
        guard let cur = currentInnings else { return "—" }
        let name = match.teamName(isTeamA: cur.battingTeamIsA)
        if cur.deliveries.isEmpty {
            if cur.order == 1 {
                let winner = match.teamName(isTeamA: match.tossWinnerIsA)
                return "\(winner) won the toss, chose to \(match.tossDecision == .bat ? "bat" : "bowl")"
            }
            return "Innings break — \(name) to bat"
        }
        return cur.order == 2 ? "\(name) chasing" : "\(name) batting"
    }

    private var infoLine: String? {
        guard let cur = liveInnings else { return nil }
        let s = MatchScoring.state(for: cur, in: match)
        var parts = ["CRR \(fmt(MatchScoring.currentRunRate(s)))"]
        if cur.order == 2, let target = MatchScoring.target(for: cur, in: match) {
            let need = max(0, target - s.totalRuns)
            let ballsLeft = match.oversPerInnings * match.ballsPerOver
                - (s.oversCompleted * match.ballsPerOver + s.ballsThisOver)
            if ballsLeft > 0 {
                parts.append("Need \(need) off \(ballsLeft)")
                if let rrr = MatchScoring.requiredRunRate(s, match: match, target: target) {
                    parts.append("RRR \(fmt(rrr))")
                }
            }
        }
        return parts.joined(separator: " · ")
    }

    private func fmt(_ v: Double) -> String { String(format: "%.2f", v) }

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
