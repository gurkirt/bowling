//
//  MiniScoreboard.swift
//  CricReel
//
//  Compact live scoreboard: score/target, run rates, both batters, current + last bowler.
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
        VStack(spacing: 12) {
            header
            ratesRow
            Divider()
            battersTable
            Divider()
            bowlersTable
        }
        .padding(16)
        .background(.background, in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(Color(.separator).opacity(0.4)))
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(battingName).font(.subheadline).foregroundStyle(.secondary)
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(state.totalRuns)/\(state.wickets)")
                        .font(.system(size: 34, weight: .bold, design: .rounded)).monospacedDigit()
                    Text("(\(state.oversDisplay)/\(match.oversPerInnings))")
                        .font(.subheadline).foregroundStyle(.secondary).monospacedDigit()
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                if let t = target {
                    Text("Target").font(.caption).foregroundStyle(.secondary)
                    Text("\(t)").font(.title3.bold()).monospacedDigit()
                } else {
                    Text("1st Innings").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var ratesRow: some View {
        HStack {
            Text("CRR \(fmt(MatchScoring.currentRunRate(state)))")
                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            Spacer()
            if let t = target,
               let rrr = MatchScoring.requiredRunRate(state, match: match, target: t) {
                let need = max(0, t - state.totalRuns)
                let ballsLeft = match.oversPerInnings * match.ballsPerOver
                    - (state.oversCompleted * match.ballsPerOver + state.ballsThisOver)
                Text("Need \(need) off \(ballsLeft)  •  RRR \(fmt(rrr))")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        }
    }

    // MARK: - Batters

    private var battersTable: some View {
        Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 6) {
            GridRow {
                Text("Batter").gridColumnAlignment(.leading)
                Text("R").gridColumnAlignment(.trailing)
                Text("B").gridColumnAlignment(.trailing)
                Text("4s").gridColumnAlignment(.trailing)
                Text("6s").gridColumnAlignment(.trailing)
                Text("SR").gridColumnAlignment(.trailing)
            }
            .font(.caption2).foregroundStyle(.secondary)

            batterRow(state.strikerID, onStrike: true)
            batterRow(state.nonStrikerID, onStrike: false)
        }
    }

    private func batterRow(_ id: UUID?, onStrike: Bool) -> some View {
        let line = id.flatMap { battingLines[$0] }
        return GridRow {
            HStack(spacing: 4) {
                Image(systemName: onStrike ? "circle.fill" : "circle")
                    .font(.system(size: 7)).foregroundStyle(onStrike ? .green : .secondary)
                Text(lookup.name(id)).lineLimit(1)
            }
            num(line?.runs ?? 0)
            num(line?.ballsFaced ?? 0)
            num(line?.fours ?? 0)
            num(line?.sixes ?? 0)
            Text(fmt(line?.strikeRate ?? 0)).font(.callout).monospacedDigit()
                .gridColumnAlignment(.trailing)
        }
        .font(onStrike ? .callout.bold() : .callout)
    }

    // MARK: - Bowlers

    private var bowlersTable: some View {
        Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 6) {
            GridRow {
                Text("Bowler").gridColumnAlignment(.leading)
                Text("O").gridColumnAlignment(.trailing)
                Text("R").gridColumnAlignment(.trailing)
                Text("W").gridColumnAlignment(.trailing)
                Text("Econ").gridColumnAlignment(.trailing)
            }
            .font(.caption2).foregroundStyle(.secondary)

            bowlerRow(selectedBowlerID, label: "current")
            if lastBowlerID != nil, lastBowlerID != selectedBowlerID {
                bowlerRow(lastBowlerID, label: "last")
            }
        }
    }

    private func bowlerRow(_ id: UUID?, label: String) -> some View {
        let line = id.flatMap { bowlingLines[$0] }
        return GridRow {
            Text(lookup.name(id)).lineLimit(1)
            Text(line?.oversDisplay ?? "0.0").monospacedDigit().gridColumnAlignment(.trailing)
            num(line?.runsConceded ?? 0)
            num(line?.wickets ?? 0)
            Text(fmt(line?.economy ?? 0)).monospacedDigit().gridColumnAlignment(.trailing)
        }
        .font(label == "current" ? .callout.bold() : .callout)
        .foregroundStyle(label == "current" ? .primary : .secondary)
    }

    // MARK: - Helpers

    private func num(_ value: Int) -> some View {
        Text("\(value)").font(.callout).monospacedDigit().gridColumnAlignment(.trailing)
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
