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
        VStack(alignment: .leading, spacing: 7) {
            teamsLine
            Divider()
            battersLine
            bowlersLine
            if let result = matchResult {
                Label(result, systemImage: "trophy.fill")
                    .font(.caption).foregroundStyle(.orange)
            } else {
                ratesLine
            }
        }
        .padding(12)
        .background(.background, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color(.separator).opacity(0.35)))
    }

    // MARK: - Teams
    // Left:  IND 45/2 (3.4) · T 121 (12.0)   Right: AUS 120/6

    private var teamsLine: some View {
        let batName = match.teamReelName(isTeamA: innings.battingTeamIsA)
        let otherName = match.teamReelName(isTeamA: !innings.battingTeamIsA)
        let other = MatchScoring.teamScore(isTeamA: !innings.battingTeamIsA, in: match)
        // Left side: batting score + overs, plus chase target if applicable
        let leftSuffix = target.map { " · T \($0) (\(match.oversPerInnings).0)" } ?? ""
        let leftText = "\(batName) \(state.totalRuns)/\(state.wickets) (\(state.oversDisplay))\(leftSuffix)"
        // Right side: always the opponent (score if they have batted, otherwise just name)
        let rightText: String = other.batted
            ? "\(otherName) \(other.runs)/\(other.wickets)"
            : otherName
        return HStack(spacing: 0) {
            HStack(spacing: 5) {
                if !state.isInningsComplete {
                    Circle().fill(.green).frame(width: 7, height: 7)
                }
                Text(leftText)
                    .font(.headline).monospacedDigit()
            }
            Spacer(minLength: 12)
            Text(rightText)
                .font(.subheadline.weight(.medium)).monospacedDigit()
                .foregroundStyle(.secondary)
        }
        .lineLimit(1)
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

    // MARK: - Batter / bowler lines (both players in one line each)

    private var battersLine: some View {
        HStack(spacing: 0) {
            HStack(spacing: 5) {
                Image(systemName: "figure.cricket").font(.caption2).foregroundStyle(.secondary)
                batterText(state.strikerID, onStrike: true)
            }
            Spacer(minLength: 12)
            batterText(state.nonStrikerID, onStrike: false)
        }
        .font(.subheadline).monospacedDigit().lineLimit(1)
    }

    private func batterText(_ id: UUID?, onStrike: Bool) -> some View {
        let line = id.flatMap { battingLines[$0] }
        return Text("\(lookup.reel(id))\(onStrike ? "*" : "") \(line?.runs ?? 0)(\(line?.ballsFaced ?? 0))")
            .fontWeight(onStrike ? .semibold : .regular)
            .foregroundStyle(onStrike ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
    }

    private var bowlersLine: some View {
        HStack(spacing: 0) {
            HStack(spacing: 5) {
                Image(systemName: "cricket.ball").font(.caption2).foregroundStyle(.secondary)
                bowlerText(selectedBowlerID, isCurrent: true)
            }
            if let last = lastBowlerID, last != selectedBowlerID {
                Spacer(minLength: 12)
                bowlerText(last, isCurrent: false)
            } else {
                Spacer(minLength: 0)
            }
        }
        .font(.subheadline).monospacedDigit().lineLimit(1)
    }

    private func bowlerText(_ id: UUID?, isCurrent: Bool) -> some View {
        let line = id.flatMap { bowlingLines[$0] }
        return Text("\(lookup.reel(id)) \(line?.wickets ?? 0)–\(line?.runsConceded ?? 0) (\(line?.oversDisplay ?? "0.0"))")
            .fontWeight(isCurrent ? .semibold : .regular)
            .foregroundStyle(isCurrent ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
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
