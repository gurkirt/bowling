//
//  MatchScoring.swift
//  CricReel
//
//  Bridges SwiftData models to the pure ScoringEngine: batting order, rules, derived
//  state, chase target and run rates.
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

    /// First-innings total, if the first innings exists.
    static func firstInningsTotal(in match: Match) -> Int? {
        guard let first = match.innings.first(where: { $0.order == 1 }) else { return nil }
        return state(for: first, in: match, target: nil).totalRuns
    }

    /// Target for the given innings (only the second innings chases).
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
            rules: rules(for: match, innings: innings, target: target))
    }

    static func appearedBatters(for innings: Innings, in match: Match) -> Set<UUID> {
        ScoringEngine.appearedBatters(
            battingOrder: battingOrder(for: innings, in: match),
            deliveries: deliveryData(for: innings))
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
}
