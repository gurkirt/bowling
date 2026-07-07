//
//  ScorecardView.swift
//  CricReel
//
//  Full scorecard per innings: batting and bowling tables derived from deliveries.
//

import SwiftUI
import SwiftData

struct ScorecardView: View {
    @Bindable var match: Match
    @Query(sort: \Player.name) private var players: [Player]

    private var lookup: PlayerLookup { PlayerLookup(players) }

    var body: some View {
        List {
            ForEach(match.innings.sorted { $0.order < $1.order }) { innings in
                inningsSection(innings)
            }
        }
        .navigationTitle("Scorecard")
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder
    private func inningsSection(_ innings: Innings) -> some View {
        let deliveries = MatchScoring.deliveryData(for: innings)
        let state = MatchScoring.state(for: innings, in: match)
        let batting = StatsBuilder.batting(from: deliveries)
        let bowling = StatsBuilder.bowling(from: deliveries)
        let battingName = innings.battingTeamIsA ? match.teamAName : match.teamBName
        let battingOrder = MatchScoring.battingOrder(for: innings, in: match)
        let bowlingIDs = orderedBowlerIDs(deliveries)
        let extras = deliveries.reduce(0) { $0 + $1.extraRuns }

        Section {
            HStack {
                Text(battingName).font(.headline)
                Spacer()
                Text("\(state.totalRuns)/\(state.wickets)").bold().monospacedDigit()
                Text("(\(state.oversDisplay))").foregroundStyle(.secondary).monospacedDigit()
            }
        }

        Section("Batting") {
            HStack {
                Text("Batter").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("R  B  4s 6s  SR").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            ForEach(battingOrder.filter { batting[$0] != nil }, id: \.self) { id in
                if let line = batting[id] {
                    battingRow(name: lookup.name(id), line: line)
                }
            }
            HStack {
                Text("Extras").foregroundStyle(.secondary)
                Spacer()
                Text("\(extras)").monospacedDigit()
            }
            HStack {
                Text("Total").bold()
                Spacer()
                Text("\(state.totalRuns)/\(state.wickets)").bold().monospacedDigit()
            }
        }

        Section("Bowling") {
            HStack {
                Text("Bowler").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("O   R   W   Econ").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            ForEach(bowlingIDs, id: \.self) { id in
                if let line = bowling[id] {
                    bowlingRow(name: lookup.name(id), line: line)
                }
            }
        }
    }

    private func battingRow(name: String, line: BattingLine) -> some View {
        HStack {
            VStack(alignment: .leading) {
                Text(name)
                if line.isOut, let d = line.dismissal {
                    Text(d.displayName).font(.caption2).foregroundStyle(.secondary)
                } else {
                    Text("not out").font(.caption2).foregroundStyle(.secondary)
                }
            }
            Spacer()
            Text("\(line.runs)  \(line.ballsFaced)  \(line.fours)  \(line.sixes)  \(srText(line.strikeRate))")
                .monospacedDigit().font(.callout)
        }
    }

    private func bowlingRow(name: String, line: BowlingLine) -> some View {
        HStack {
            Text(name)
            Spacer()
            Text("\(line.oversDisplay)  \(line.runsConceded)  \(line.wickets)  \(srText(line.economy))")
                .monospacedDigit().font(.callout)
        }
    }

    private func srText(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    /// Bowlers in the order they first bowled.
    private func orderedBowlerIDs(_ deliveries: [DeliveryData]) -> [UUID] {
        var seen: [UUID] = []
        for d in deliveries where !seen.contains(d.bowlerID) {
            seen.append(d.bowlerID)
        }
        return seen
    }
}
