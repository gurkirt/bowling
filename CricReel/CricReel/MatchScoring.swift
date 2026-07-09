//
//  MatchScoring.swift
//  CricReel
//
//  Bridges SwiftData models to the pure ScoringEngine: batting order, openers, rules,
//  derived state, chase target, run rates and bowler quota.
//

import Foundation

enum MatchScoring {

    static func battingOrder(for innings: Innings, in match: Match) -> [UUID] {
        match.lineupIDs(isTeamA: innings.battingTeamIsA)
    }

    static func bowlingOrder(for innings: Innings, in match: Match) -> [UUID] {
        match.lineupIDs(isTeamA: !innings.battingTeamIsA)
    }

    static func deliveryData(for innings: Innings) -> [DeliveryData] {
        innings.orderedDeliveries.map(DeliveryData.init)
    }

    /// Chosen openers (falls back to the first two in the batting order).
    static func openers(for innings: Innings, in match: Match) -> (striker: UUID, nonStriker: UUID?)? {
        let order = battingOrder(for: innings, in: match)
        if let s = innings.openerStrikerID {
            return (s, innings.openerNonStrikerID)
        }
        guard let first = order.first else { return nil }
        return (first, order.count > 1 ? order[1] : nil)
    }

    /// Whether openers still need to be chosen for this innings.
    static func needsOpeners(_ innings: Innings) -> Bool {
        innings.openerStrikerID == nil && innings.deliveries.isEmpty
    }

    static func firstInningsTotal(in match: Match) -> Int? {
        guard let first = match.innings.first(where: { $0.order == 1 }) else { return nil }
        return state(for: first, in: match, target: nil).totalRuns
    }

    static func target(for innings: Innings, in match: Match) -> Int? {
        guard innings.order == 2, let t = firstInningsTotal(in: match) else { return nil }
        return t + 1
    }

    static func rules(for match: Match, innings: Innings, target: Int?) -> InningsRules {
        InningsRules(
            oversLimit: match.oversPerInnings,
            ballsPerOver: match.ballsPerOver,
            runsPerWide: match.runsPerWide,
            runsPerNoBall: match.runsPerNoBall,
            wideIsLegalBall: match.wideIsLegalBall,
            noBallIsLegalBall: match.noBallIsLegalBall,
            lineupSize: max(2, match.playersPerSide),
            target: target)
    }

    static func rules(for match: Match, innings: Innings) -> InningsRules {
        rules(for: match, innings: innings, target: target(for: innings, in: match))
    }

    static func state(for innings: Innings, in match: Match) -> InningsState {
        state(for: innings, in: match, target: target(for: innings, in: match))
    }

    static func state(for innings: Innings, in match: Match, target: Int?) -> InningsState {
        ScoringEngine.computeState(
            battingOrder: battingOrder(for: innings, in: match),
            deliveries: deliveryData(for: innings),
            rules: rules(for: match, innings: innings, target: target),
            openers: openers(for: innings, in: match))
    }

    /// State computed from an explicit delivery list (used by undo, where the SwiftData
    /// relationship may not reflect a just-deleted delivery synchronously).
    static func state(for innings: Innings, in match: Match, deliveries: [Delivery]) -> InningsState {
        ScoringEngine.computeState(
            battingOrder: battingOrder(for: innings, in: match),
            deliveries: deliveries.map(DeliveryData.init),
            rules: rules(for: match, innings: innings),
            openers: openers(for: innings, in: match))
    }

    static func appearedBatters(for innings: Innings, in match: Match) -> Set<UUID> {
        ScoringEngine.appearedBatters(
            battingOrder: battingOrder(for: innings, in: match),
            deliveries: deliveryData(for: innings),
            openers: openers(for: innings, in: match))
    }

    // MARK: - Bowler quota

    /// Overs bowled (distinct over indices) per bowler in this innings.
    static func oversBowled(for innings: Innings) -> [UUID: Int] {
        var overs: [UUID: Set<Int>] = [:]
        for d in innings.orderedDeliveries {
            overs[d.bowlerID, default: []].insert(d.overNumber)
        }
        return overs.mapValues { $0.count }
    }

    /// Bowlers eligible to bowl the next over: on the bowling side, not the bowler who
    /// bowled the previous over, and with quota remaining (if a max is set).
    static func eligibleBowlers(for innings: Innings, in match: Match, excluding lastBowler: UUID?) -> [UUID] {
        let overs = oversBowled(for: innings)
        let maxOvers = match.maxOversPerBowler
        return bowlingOrder(for: innings, in: match).filter { id in
            if id == lastBowler { return false }
            if maxOvers > 0, (overs[id] ?? 0) >= maxOvers { return false }
            return true
        }
    }

    // MARK: - Run rates

    static func currentRunRate(_ state: InningsState) -> Double {
        state.oversAsDouble == 0 ? 0 : Double(state.totalRuns) / state.oversAsDouble
    }

    static func requiredRunRate(_ state: InningsState, match: Match, target: Int?) -> Double? {
        guard let target else { return nil }
        let ballsBowled = state.oversCompleted * match.ballsPerOver + state.ballsThisOver
        let ballsRemaining = match.oversPerInnings * match.ballsPerOver - ballsBowled
        guard ballsRemaining > 0 else { return nil }
        let runsNeeded = max(0, target - state.totalRuns)
        return Double(runsNeeded) / (Double(ballsRemaining) / Double(match.ballsPerOver))
    }

    // MARK: - Team score summaries

    /// (runs, wickets, oversDisplay, hasBatted) for the given side.
    static func teamScore(isTeamA: Bool, in match: Match) -> (runs: Int, wickets: Int, overs: String, batted: Bool) {
        guard let innings = match.innings.first(where: { $0.battingTeamIsA == isTeamA }),
              !innings.deliveries.isEmpty else {
            return (0, 0, "0.0", false)
        }
        let s = state(for: innings, in: match)
        return (s.totalRuns, s.wickets, s.oversDisplay, true)
    }
}
