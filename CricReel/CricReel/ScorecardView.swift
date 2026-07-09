//
//  ScorecardView.swift
//  CricReel
//
//  Full scorecard styled after Cricinfo: a match-summary header box, aligned batting
//  and bowling tables, extras breakdown, did-not-bat / yet-to-bat, and fall of wickets.
//

import SwiftUI
import SwiftData

struct ScorecardView: View {
    @Bindable var match: Match
    @Query(sort: \Player.name) private var players: [Player]

    private var lookup: PlayerLookup { PlayerLookup(players) }

    var body: some View {
        List {
            summarySection
            ForEach(match.innings.sorted { $0.order < $1.order }) { innings in
                if !innings.deliveries.isEmpty || innings.openerStrikerID != nil {
                    battingSection(innings)
                    bowlingSection(innings)
                }
            }
        }
        .navigationTitle("Scorecard")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Summary header box

    private var summarySection: some View {
        Section {
            VStack(spacing: 8) {
                teamSummaryRow(isTeamA: true)
                teamSummaryRow(isTeamA: false)
                Divider()
                Text(contextLine)
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.vertical, 2)
        }
    }

    private func teamSummaryRow(isTeamA: Bool) -> some View {
        let s = MatchScoring.teamScore(isTeamA: isTeamA, in: match)
        return HStack {
            Text(match.teamName(isTeamA: isTeamA)).font(.headline)
            Spacer()
            if s.batted {
                Text("\(s.runs)/\(s.wickets)").font(.headline).monospacedDigit()
                Text("(\(s.overs) ov)").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            } else {
                Text("yet to bat").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var contextLine: String {
        if let r = matchResult { return r }
        let winner = match.teamName(isTeamA: match.tossWinnerIsA)
        return "\(winner) won the toss and elected to \(match.tossDecision == .bat ? "bat" : "bowl") first"
    }

    private var matchResult: String? {
        guard match.status == .completed,
              let first = match.innings.first(where: { $0.order == 1 }),
              let second = match.innings.first(where: { $0.order == 2 }) else { return nil }
        let s1 = MatchScoring.state(for: first, in: match)
        let s2 = MatchScoring.state(for: second, in: match)
        let firstName = match.teamName(isTeamA: first.battingTeamIsA)
        let secondName = match.teamName(isTeamA: second.battingTeamIsA)
        if s2.totalRuns > s1.totalRuns {
            let wktsLeft = max(0, match.playersPerSide - 1 - s2.wickets)
            return "\(secondName) won by \(wktsLeft) wicket\(wktsLeft == 1 ? "" : "s")"
        }
        if s1.totalRuns > s2.totalRuns {
            let by = s1.totalRuns - s2.totalRuns
            return "\(firstName) won by \(by) run\(by == 1 ? "" : "s")"
        }
        return "Match tied"
    }

    // MARK: - Batting

    @ViewBuilder
    private func battingSection(_ innings: Innings) -> some View {
        let deliveries = innings.orderedDeliveries
        let state = MatchScoring.state(for: innings, in: match)
        let batting = StatsBuilder.batting(from: deliveries)
        let order = battedOrder(innings)
        let dnb = didNotBat(innings)
        let fow = fallOfWickets(deliveries)
        let extras = extrasBreakdown(deliveries)
        let teamName = match.teamName(isTeamA: innings.battingTeamIsA)

        Section("\(teamName) — \(state.totalRuns)/\(state.wickets) (\(state.oversDisplay) ov)") {
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    Text("Batter").gridColumnAlignment(.leading)
                    Text("R").gridColumnAlignment(.trailing)
                    Text("B").gridColumnAlignment(.trailing)
                    Text("4s").gridColumnAlignment(.trailing)
                    Text("6s").gridColumnAlignment(.trailing)
                    Text("SR").gridColumnAlignment(.trailing)
                }
                .font(.caption).foregroundStyle(.secondary)

                ForEach(order, id: \.self) { id in
                    let line = batting[id] ?? BattingLine(playerID: id)
                    GridRow {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(lookup.name(id)).font(.subheadline).lineLimit(1)
                            Text(dismissalTextFor(id, deliveries: deliveries))
                                .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                        }
                        num("\(line.runs)", bold: true)
                        num("\(line.ballsFaced)")
                        num("\(line.fours)")
                        num("\(line.sixes)")
                        num(fmt(line.strikeRate))
                    }
                }
            }

            HStack {
                Text("Extras").foregroundStyle(.secondary)
                if !extras.detail.isEmpty {
                    Text(extras.detail).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(extras.total)").monospacedDigit()
            }
            .font(.subheadline)

            HStack {
                Text("Total").bold()
                Spacer()
                Text("\(state.totalRuns)/\(state.wickets)").bold().monospacedDigit()
                Text("(\(state.oversDisplay) Ov, RR \(fmt(MatchScoring.currentRunRate(state))))")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            .font(.subheadline)

            if !dnb.isEmpty {
                labelledList(innings.isComplete ? "Did not bat" : "Yet to bat",
                             dnb.map { lookup.name($0) }.joined(separator: ", "))
            }
            if !fow.isEmpty {
                labelledList("Fall of wickets",
                             fow.map { "\($0.wkt)-\($0.runs) (\($0.name), \($0.over))" }.joined(separator: ", "))
            }
        }
    }

    // MARK: - Bowling

    @ViewBuilder
    private func bowlingSection(_ innings: Innings) -> some View {
        let deliveries = innings.orderedDeliveries
        let bowling = StatsBuilder.bowling(from: deliveries)
        let ids = orderedBowlerIDs(deliveries)
        if !ids.isEmpty {
            Section("Bowling") {
                Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 8) {
                    GridRow {
                        Text("Bowler").gridColumnAlignment(.leading)
                        Text("O").gridColumnAlignment(.trailing)
                        Text("M").gridColumnAlignment(.trailing)
                        Text("R").gridColumnAlignment(.trailing)
                        Text("W").gridColumnAlignment(.trailing)
                        Text("Econ").gridColumnAlignment(.trailing)
                    }
                    .font(.caption).foregroundStyle(.secondary)

                    ForEach(ids, id: \.self) { id in
                        let line = bowling[id] ?? BowlingLine(playerID: id)
                        GridRow {
                            Text(lookup.name(id)).font(.subheadline).lineLimit(1)
                            num(line.oversDisplay)
                            num("\(line.maidens)")
                            num("\(line.runsConceded)")
                            num("\(line.wickets)", bold: true)
                            num(fmt(line.economy))
                        }
                    }
                }
            }
        }
    }

    // MARK: - Row helpers

    private func num(_ s: String, bold: Bool = false) -> some View {
        Text(s)
            .font(.subheadline)
            .fontWeight(bold ? .bold : .regular)
            .monospacedDigit()
            .gridColumnAlignment(.trailing)
    }

    private func labelledList(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased()).font(.caption2).bold().foregroundStyle(.secondary)
            Text(body).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func fmt(_ v: Double) -> String { String(format: "%.2f", v) }

    // MARK: - Derivations

    /// Batters in the order they came to the crease (openers, then each new batter).
    private func battedOrder(_ innings: Innings) -> [UUID] {
        let order = MatchScoring.battingOrder(for: innings, in: match)
        var result: [UUID] = []
        if let o = MatchScoring.openers(for: innings, in: match) {
            result.append(o.striker)
            if let n = o.nonStriker { result.append(n) }
        }
        for d in innings.orderedDeliveries {
            if let nb = d.newBatterID, !result.contains(nb) { result.append(nb) }
        }
        return result.filter { order.contains($0) }
    }

    private func didNotBat(_ innings: Innings) -> [UUID] {
        let appeared = Set(battedOrder(innings))
        return MatchScoring.battingOrder(for: innings, in: match).filter { !appeared.contains($0) }
    }

    /// Dismissal text for a batter, or "not out".
    private func dismissalTextFor(_ id: UUID, deliveries: [Delivery]) -> String {
        if let wicket = deliveries.first(where: { $0.isWicket && ($0.dismissedPlayerID ?? $0.strikerID) == id }) {
            return DeliveryFormatting.dismissalText(wicket, lookup: lookup)
        }
        return "not out"
    }

    private struct FoW { let wkt: Int; let runs: Int; let name: String; let over: String }

    private func fallOfWickets(_ deliveries: [Delivery]) -> [FoW] {
        var cum = 0, wkt = 0
        var result: [FoW] = []
        for d in deliveries {
            cum += d.totalRuns
            if d.isWicket {
                wkt += 1
                let name = lookup.name(d.dismissedPlayerID ?? d.strikerID)
                result.append(FoW(wkt: wkt, runs: cum, name: name, over: "\(d.overNumber).\(d.ballInOver)"))
            }
        }
        return result
    }

    private func extrasBreakdown(_ deliveries: [Delivery]) -> (total: Int, detail: String) {
        var b = 0, lb = 0, w = 0, nb = 0
        for d in deliveries {
            switch d.extraType {
            case .wide:   w += d.extraRuns
            case .noBall: nb += d.extraRuns
            case .bye:    b += d.extraRuns
            case .legBye: lb += d.extraRuns
            case .none:   break
            }
        }
        var parts: [String] = []
        if b > 0 { parts.append("b \(b)") }
        if lb > 0 { parts.append("lb \(lb)") }
        if w > 0 { parts.append("w \(w)") }
        if nb > 0 { parts.append("nb \(nb)") }
        return (b + lb + w + nb, parts.isEmpty ? "" : "(\(parts.joined(separator: ", ")))")
    }

    private func orderedBowlerIDs(_ deliveries: [Delivery]) -> [UUID] {
        var seen: [UUID] = []
        for d in deliveries where !seen.contains(d.bowlerID) { seen.append(d.bowlerID) }
        return seen
    }
}
