//
//  MiniScoreboard.swift
//  CricReel
//
//  Compact live scoreboard: both team scores, result/toss line, both batters (runs &
//  balls) and current + last bowler (w–r, overs). Uses reel names.
//

import SwiftUI

struct MiniScoreboard: View {
    let match: Match
    let innings: Innings
    let state: InningsState
    let target: Int?
    let selectedBowlerID: UUID?
    let lastBowlerID: UUID?
    let battingLines: [UUID: BattingLine]
    let bowlingLines: [UUID: BowlingLine]
    let lookup: PlayerLookup

    var body: some View {
        VStack(spacing: 8) {
            teamRow(isTeamA: true)
            teamRow(isTeamA: false)
            contextLine
            Divider()
            batterRow(state.strikerID, onStrike: true)
            batterRow(state.nonStrikerID, onStrike: false)
            Divider()
            bowlerRow(selectedBowlerID, isCurrent: true)
            if let last = lastBowlerID, last != selectedBowlerID {
                bowlerRow(last, isCurrent: false)
            }
            ratesLine
        }
        .padding(14)
        .background(.background, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color(.separator).opacity(0.35)))
    }

    // MARK: - Team rows

    private func teamRow(isTeamA: Bool) -> some View {
        let s = MatchScoring.teamScore(isTeamA: isTeamA, in: match)
        let isBatting = innings.battingTeamIsA == isTeamA && !state.isInningsComplete
        return HStack {
            HStack(spacing: 6) {
                Circle().fill(isBatting ? Color.green : .clear)
                    .frame(width: 7, height: 7)
                Text(match.teamReelName(isTeamA: isTeamA))
                    .font(.headline)
            }
            Spacer()
            if s.batted {
                Text("\(s.runs)/\(s.wickets)")
                    .font(.title3.bold()).monospacedDigit()
                Text("(\(s.overs))")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            } else {
                Text("yet to bat").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var contextLine: some View {
        Group {
            if let result = matchResult {
                Label(result, systemImage: "trophy.fill").foregroundStyle(.orange)
            } else {
                Text(tossLine).foregroundStyle(.secondary)
            }
        }
        .font(.caption)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var tossLine: String {
        let winner = match.teamReelName(isTeamA: match.tossWinnerIsA)
        return "\(winner) won the toss & chose to \(match.tossDecision == .bat ? "bat" : "bowl")"
    }

    private var matchResult: String? {
        guard match.status == .completed,
              let first = match.innings.first(where: { $0.order == 1 }),
              let second = match.innings.first(where: { $0.order == 2 }) else { return nil }
        let s1 = MatchScoring.state(for: first, in: match)
        let s2 = MatchScoring.state(for: second, in: match)
        let firstName = match.teamReelName(isTeamA: first.battingTeamIsA)
        let secondName = match.teamReelName(isTeamA: second.battingTeamIsA)
        if s2.totalRuns > s1.totalRuns {
            let wktsLeft = max(0, match.playersPerSide - 1 - s2.wickets)
            return "\(secondName) won by \(wktsLeft) wkt\(wktsLeft == 1 ? "" : "s")"
        }
        if s1.totalRuns > s2.totalRuns {
            let by = s1.totalRuns - s2.totalRuns
            return "\(firstName) won by \(by) run\(by == 1 ? "" : "s")"
        }
        return "Match tied"
    }

    // MARK: - Batter / bowler rows

    private func batterRow(_ id: UUID?, onStrike: Bool) -> some View {
        let line = id.flatMap { battingLines[$0] }
        return HStack(spacing: 8) {
            Image(systemName: onStrike ? "circle.fill" : "circle")
                .font(.system(size: 7)).foregroundStyle(onStrike ? .green : .secondary)
            Text(lookup.reel(id)).lineLimit(1)
            Spacer()
            Text("\(line?.runs ?? 0) (\(line?.ballsFaced ?? 0))").monospacedDigit()
        }
        .font(onStrike ? .subheadline.bold() : .subheadline)
    }

    private func bowlerRow(_ id: UUID?, isCurrent: Bool) -> some View {
        let line = id.flatMap { bowlingLines[$0] }
        return HStack(spacing: 8) {
            Image(systemName: "cricket.ball").font(.caption2)
                .foregroundStyle(isCurrent ? .primary : .secondary)
            Text(lookup.reel(id)).lineLimit(1)
            Spacer()
            Text("\(line?.wickets ?? 0)–\(line?.runsConceded ?? 0) (\(line?.oversDisplay ?? "0.0"))")
                .monospacedDigit()
        }
        .font(isCurrent ? .subheadline.bold() : .subheadline)
        .foregroundStyle(isCurrent ? .primary : .secondary)
    }

    private var ratesLine: some View {
        HStack {
            Text("CRR \(fmt(MatchScoring.currentRunRate(state)))")
            Spacer()
            if let t = target {
                let need = max(0, t - state.totalRuns)
                let ballsLeft = match.oversPerInnings * match.ballsPerOver
                    - (state.oversCompleted * match.ballsPerOver + state.ballsThisOver)
                let rrr = MatchScoring.requiredRunRate(state, match: match, target: t)
                Text("Need \(need) off \(ballsLeft)" + (rrr.map { " · RRR \(fmt($0))" } ?? ""))
            }
        }
        .font(.caption2).foregroundStyle(.secondary).monospacedDigit()
    }

    private func fmt(_ value: Double) -> String { String(format: "%.2f", value) }
}

struct BowlerPickerSheet: View {
    let bowlerIDs: [UUID]
    let lookup: PlayerLookup
    let current: UUID?
    var onSelect: (UUID) -> Void
    var onUndo: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if let onUndo {
                    Section {
                        Button(role: .destructive) { onUndo() } label: {
                            Label("Undo Last Ball", systemImage: "arrow.uturn.backward")
                        }
                    } footer: {
                        Text("Removes the previous delivery and reopens that over.")
                    }
                }
                Section {
                    if bowlerIDs.isEmpty {
                        Text("No eligible bowlers (quota reached).").foregroundStyle(.secondary)
                    }
                    ForEach(bowlerIDs, id: \.self) { id in
                        Button {
                            onSelect(id); dismiss()
                        } label: {
                            HStack {
                                Text(lookup.name(id)).foregroundStyle(.primary)
                                Spacer()
                                if id == current { Image(systemName: "checkmark").foregroundStyle(.blue) }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Select Bowler")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
        }
    }
}
