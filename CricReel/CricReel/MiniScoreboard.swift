//
//  MiniScoreboard.swift
//  CricReel
//
//  Compact live scoreboard: score/target, run rates, both batters (runs & balls),
//  and current + last bowler (w–r, overs).
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

    private var battingName: String { innings.battingTeamIsA ? match.teamAName : match.teamBName }

    var body: some View {
        VStack(spacing: 10) {
            header
            if let t = target { chaseRow(t) } else { crrRow }
            Divider()
            batterRow(state.strikerID, onStrike: true)
            batterRow(state.nonStrikerID, onStrike: false)
            Divider()
            bowlerRow(selectedBowlerID, isCurrent: true)
            if let last = lastBowlerID, last != selectedBowlerID {
                bowlerRow(last, isCurrent: false)
            }
        }
        .padding(14)
        .background(.background, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color(.separator).opacity(0.35)))
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(battingName).font(.headline)
            Text("\(state.totalRuns)/\(state.wickets)")
                .font(.system(size: 28, weight: .bold, design: .rounded)).monospacedDigit()
            Text("(\(state.oversDisplay)/\(match.oversPerInnings))")
                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            Spacer()
            if let t = target {
                Text("Target \(t)").font(.subheadline.bold()).foregroundStyle(.secondary).monospacedDigit()
            } else {
                Text("1st Innings").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var crrRow: some View {
        HStack {
            Text("CRR \(fmt(MatchScoring.currentRunRate(state)))")
                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            Spacer()
        }
    }

    private func chaseRow(_ t: Int) -> some View {
        let need = max(0, t - state.totalRuns)
        let ballsLeft = match.oversPerInnings * match.ballsPerOver
            - (state.oversCompleted * match.ballsPerOver + state.ballsThisOver)
        let rrr = MatchScoring.requiredRunRate(state, match: match, target: t)
        return HStack {
            Text("CRR \(fmt(MatchScoring.currentRunRate(state)))")
            Spacer()
            Text("Need \(need) off \(ballsLeft)" + (rrr.map { " · RRR \(fmt($0))" } ?? ""))
        }
        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
    }

    private func batterRow(_ id: UUID?, onStrike: Bool) -> some View {
        let line = id.flatMap { battingLines[$0] }
        return HStack(spacing: 8) {
            Image(systemName: onStrike ? "circle.fill" : "circle")
                .font(.system(size: 7)).foregroundStyle(onStrike ? .green : .secondary)
            Text(lookup.name(id)).lineLimit(1)
            Spacer()
            Text("\(line?.runs ?? 0) (\(line?.ballsFaced ?? 0))").monospacedDigit()
        }
        .font(onStrike ? .subheadline.bold() : .subheadline)
    }

    private func bowlerRow(_ id: UUID?, isCurrent: Bool) -> some View {
        let line = id.flatMap { bowlingLines[$0] }
        return HStack(spacing: 8) {
            Image(systemName: "figure.cricket").font(.caption2)
                .foregroundStyle(isCurrent ? .primary : .secondary)
            Text(lookup.name(id)).lineLimit(1)
            Spacer()
            Text("\(line?.wickets ?? 0)–\(line?.runsConceded ?? 0) (\(line?.oversDisplay ?? "0.0"))")
                .monospacedDigit()
        }
        .font(isCurrent ? .subheadline.bold() : .subheadline)
        .foregroundStyle(isCurrent ? .primary : .secondary)
    }

    private func fmt(_ value: Double) -> String { String(format: "%.2f", value) }
}

struct BowlerPickerSheet: View {
    let bowlerIDs: [UUID]
    let lookup: PlayerLookup
    let current: UUID?
    var onSelect: (UUID) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List(bowlerIDs, id: \.self) { id in
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
            .navigationTitle("Select Bowler")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
        }
    }
}
