//
//  ScoringEngine.swift
//  CricReel
//
//  Pure, SwiftData-independent scoring logic. State is derived by replaying the
//  ordered list of deliveries, which keeps the live scoreboard and any edits/undo
//  always consistent.
//
//  v1 rules: runs 0–6, wickets, wides. Strike rotates on odd runs off the bat and
//  at the end of each (legal) over. A wide adds `runsPerWide`, does not count as a
//  legal ball, is not faced by the batter, and does not rotate strike.
//

import Foundation

/// Lightweight, value-type mirror of a `Delivery` used by the engine so the rules
/// can be unit-tested without a model container.
struct DeliveryData {
    var sequence: Int
    var strikerID: UUID
    var nonStrikerID: UUID
    var bowlerID: UUID
    var runsOffBat: Int
    var extraType: ExtraType
    var extraRuns: Int
    var isWicket: Bool
    var dismissalType: DismissalType?
    var dismissedPlayerID: UUID?
    var isLegalDelivery: Bool
}

extension DeliveryData {
    init(_ d: Delivery) {
        self.init(sequence: d.sequence,
                  strikerID: d.strikerID,
                  nonStrikerID: d.nonStrikerID,
                  bowlerID: d.bowlerID,
                  runsOffBat: d.runsOffBat,
                  extraType: d.extraType,
                  extraRuns: d.extraRuns,
                  isWicket: d.isWicket,
                  dismissalType: d.dismissalType,
                  dismissedPlayerID: d.dismissedPlayerID,
                  isLegalDelivery: d.isLegalDelivery)
    }
}

/// Immutable rules for an innings.
struct InningsRules {
    var oversLimit: Int
    var ballsPerOver: Int
    var runsPerWide: Int
    var wideIsLegalBall: Bool
    /// Number of batters in the batting side (playing XI).
    var lineupSize: Int
}

/// What the scorer chose for a single delivery, before it is turned into a `Delivery`.
struct BallInput {
    var runsOffBat: Int = 0
    var extraType: ExtraType = .none
    var isWicket: Bool = false
    var dismissalType: DismissalType? = nil
    /// nil → the striker is dismissed.
    var dismissedPlayerID: UUID? = nil
}

/// Derived state after replaying an innings' deliveries.
struct InningsState {
    var strikerID: UUID?
    var nonStrikerID: UUID?
    var totalRuns: Int = 0
    var wickets: Int = 0
    var legalBalls: Int = 0
    var oversCompleted: Int = 0
    var ballsThisOver: Int = 0
    var nextBatterIndex: Int = 0
    var isAllOut: Bool = false
    var isInningsComplete: Bool = false
    /// True when the last delivery finished an over (scorer must pick a new bowler).
    var justCompletedOver: Bool = false

    /// 0-based over index for the next delivery to be recorded.
    var nextOverNumber: Int { oversCompleted }
    /// 1-based legal-ball number the next legal delivery will take.
    var nextBallInOver: Int { ballsThisOver + 1 }
    /// Display string like "3.4" = 3 completed overs + 4 balls.
    var oversDisplay: String { "\(oversCompleted).\(ballsThisOver)" }
}

enum ScoringEngine {

    // MARK: - Extras resolution

    /// Resolve the extra runs and legality for a chosen extra type.
    static func resolveExtra(_ extra: ExtraType, rules: InningsRules) -> (extraRuns: Int, isLegal: Bool) {
        switch extra {
        case .none: return (0, true)
        case .wide: return (rules.runsPerWide, rules.wideIsLegalBall)
        }
    }

    // MARK: - State replay

    /// Replay deliveries in order to derive the current innings state.
    static func computeState(battingOrder: [UUID],
                             deliveries: [DeliveryData],
                             rules: InningsRules) -> InningsState {
        var state = InningsState()
        guard !battingOrder.isEmpty else { return state }

        state.strikerID = battingOrder.first
        state.nonStrikerID = battingOrder.count > 1 ? battingOrder[1] : nil
        state.nextBatterIndex = min(2, battingOrder.count)

        let ordered = deliveries.sorted { $0.sequence < $1.sequence }

        for d in ordered {
            state.justCompletedOver = false
            state.totalRuns += d.runsOffBat + d.extraRuns

            // Wicket: replace the dismissed batter (default striker) at their end.
            if d.isWicket {
                state.wickets += 1
                let outID = d.dismissedPlayerID ?? state.strikerID
                let incoming: UUID? = state.nextBatterIndex < battingOrder.count
                    ? battingOrder[state.nextBatterIndex] : nil
                state.nextBatterIndex += 1
                if let incoming {
                    if outID == state.strikerID { state.strikerID = incoming }
                    else if outID == state.nonStrikerID { state.nonStrikerID = incoming }
                } else {
                    // No partner available → end will be caught by all-out check.
                    if outID == state.strikerID { state.strikerID = nil }
                    else if outID == state.nonStrikerID { state.nonStrikerID = nil }
                }
            }

            // Strike rotation on odd runs off the bat.
            if d.runsOffBat % 2 == 1 {
                swapStrike(&state)
            }

            // Legal-ball / over accounting.
            if d.isLegalDelivery {
                state.legalBalls += 1
                state.ballsThisOver += 1
                if state.ballsThisOver >= rules.ballsPerOver {
                    // Over complete → rotate strike, advance over.
                    swapStrike(&state)
                    state.ballsThisOver = 0
                    state.oversCompleted += 1
                    state.justCompletedOver = true
                }
            }

            // Completion checks.
            state.isAllOut = state.wickets >= max(0, rules.lineupSize - 1)
            state.isInningsComplete = state.isAllOut || state.oversCompleted >= rules.oversLimit
            if state.isInningsComplete { break }
        }

        return state
    }

    private static func swapStrike(_ state: inout InningsState) {
        let s = state.strikerID
        state.strikerID = state.nonStrikerID
        state.nonStrikerID = s
    }

    // MARK: - Commentary

    /// Build a short commentary line for a delivery.
    static func commentary(over: Int,
                           ball: Int,
                           bowler: String,
                           striker: String,
                           input: BallInput,
                           rules: InningsRules) -> String {
        let prefix = "\(over).\(ball) \(bowler) to \(striker),"
        if input.isWicket {
            let how = input.dismissalType?.displayName ?? "Out"
            return "\(prefix) OUT! \(how)."
        }
        switch input.extraType {
        case .wide:
            let r = rules.runsPerWide
            return "\(prefix) Wide, \(r) run\(r == 1 ? "" : "s")."
        case .none:
            switch input.runsOffBat {
            case 0: return "\(prefix) no run."
            case 4: return "\(prefix) FOUR!"
            case 6: return "\(prefix) SIX!"
            case 1: return "\(prefix) 1 run."
            default: return "\(prefix) \(input.runsOffBat) runs."
            }
        }
    }

    /// Highlight tags for a delivery (used for reels).
    static func highlightTags(for input: BallInput) -> [HighlightTag] {
        var tags: [HighlightTag] = []
        if input.isWicket { tags.append(.wicket) }
        if input.extraType == .none {
            if input.runsOffBat == 4 { tags.append(.four) }
            if input.runsOffBat == 6 { tags.append(.six) }
        }
        return tags
    }
}
