//
//  MatchScoring.swift
//  CricReel
//
//  Bridges SwiftData models to the pure ScoringEngine: resolves batting order,
//  rules, and derived innings state.
//

import Foundation

enum MatchScoring {

    static func battingOrder(for innings: Innings, in match: Match) -> [UUID] {
        match.lineupIDs(isTeamA: innings.battingTeamIsA)
    }

    static func bowlingOrder(for innings: Innings, in match: Match) -> [UUID] {
        match.lineupIDs(isTeamA: !innings.battingTeamIsA)
    }

    static func rules(for match: Match, innings: Innings) -> InningsRules {
        InningsRules(
            oversLimit: match.oversPerInnings,
            ballsPerOver: match.ballsPerOver,
            runsPerWide: match.runsPerWide,
            wideIsLegalBall: match.wideIsLegalBall,
            lineupSize: battingOrder(for: innings, in: match).count)
    }

    static func deliveryData(for innings: Innings) -> [DeliveryData] {
        innings.orderedDeliveries.map(DeliveryData.init)
    }

    static func state(for innings: Innings, in match: Match) -> InningsState {
        ScoringEngine.computeState(
            battingOrder: battingOrder(for: innings, in: match),
            deliveries: deliveryData(for: innings),
            rules: rules(for: match, innings: innings))
    }
}
