//
//  ScoringEngine.swift
//  CricReel
//
//  Pure, SwiftData-independent scoring logic. State is derived by replaying the
//  ordered deliveries so the live scoreboard, undo and edits stay consistent.
//
//  Rules: runs 0–6, wickets, and extras (wide, no-ball, bye, leg-bye). Strike rotates
//  on odd physical runs and at the end of each over. Wickets (and manual corrections)
//  carry an explicit post-delivery striker/non-striker so run-outs and new-batter
//  placement are deterministic.
//

import Foundation

/// Value-type mirror of a `Delivery` for the engine (unit-testable without a container).
struct DeliveryData {
    var sequence: Int
    var strikerID: UUID
    var nonStrikerID: UUID
    var bowlerID: UUID
    var runsOffBat: Int
    var extraType: ExtraType
    var extraRuns: Int
    var physicalRuns: Int
    var isLegalDelivery: Bool
    var isWicket: Bool
    var strikerAfterID: UUID?
    var nonStrikerAfterID: UUID?
    var newBatterID: UUID?
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
                  physicalRuns: d.physicalRuns,
                  isLegalDelivery: d.isLegalDelivery,
                  isWicket: d.isWicket,
                  strikerAfterID: d.strikerAfterID,
                  nonStrikerAfterID: d.nonStrikerAfterID,
                  newBatterID: d.newBatterID)
    }
}

struct InningsRules {
    var oversLimit: Int
    var ballsPerOver: Int
    var runsPerWide: Int
    var runsPerNoBall: Int
    var wideIsLegalBall: Bool
    var noBallIsLegalBall: Bool
    /// Number of batters in the batting side.
    var lineupSize: Int
    /// Target to chase (first-innings total + 1). nil for the first innings.
    var target: Int?
}

/// What the scorer chose for a single delivery, before it becomes a `Delivery`.
struct BallInput {
    var extra: ExtraType = .none
    /// Runs from the pad: off the bat for `.none`/`.noBall`; additional runs run for `.wide`;
    /// byes for `.bye`/`.legBye`.
    var padRuns: Int = 0

    var isWicket: Bool = false
    var dismissal: DismissalType? = nil
    var dismissedPlayerID: UUID? = nil
    var fielderID: UUID? = nil
    var newBatterID: UUID? = nil
    /// Explicit post-delivery batters (set by the wicket / manual-strike flow).
    var strikerAfterID: UUID? = nil
    var nonStrikerAfterID: UUID? = nil
}

/// Fully-resolved run breakdown for a delivery.
struct ResolvedDelivery {
    var runsOffBat: Int
    var extraRuns: Int
    var physicalRuns: Int
    var bowlerChargedRuns: Int
    var isLegal: Bool
    var faced: Bool
}

struct InningsState {
    var strikerID: UUID?
    var nonStrikerID: UUID?
    var totalRuns: Int = 0
    var wickets: Int = 0
    var legalBalls: Int = 0
    var oversCompleted: Int = 0
    var ballsThisOver: Int = 0
    var isAllOut: Bool = false
    var isInningsComplete: Bool = false
    var targetReached: Bool = false
    var justCompletedOver: Bool = false

    var nextOverNumber: Int { oversCompleted }
    var nextBallInOver: Int { ballsThisOver + 1 }
    var oversDisplay: String { "\(oversCompleted).\(ballsThisOver)" }
    var oversAsDouble: Double { Double(oversCompleted) + Double(ballsThisOver) / 6.0 }
}

enum ScoringEngine {

    // MARK: - Extra resolution

    static func resolve(extra: ExtraType, padRuns runs: Int, rules: InningsRules) -> ResolvedDelivery {
        switch extra {
        case .none:
            return ResolvedDelivery(runsOffBat: runs, extraRuns: 0, physicalRuns: runs,
                                    bowlerChargedRuns: runs, isLegal: true, faced: true)
        case .wide:
            let total = rules.runsPerWide + runs
            return ResolvedDelivery(runsOffBat: 0, extraRuns: total, physicalRuns: runs,
                                    bowlerChargedRuns: total, isLegal: rules.wideIsLegalBall, faced: false)
        case .noBall:
            return ResolvedDelivery(runsOffBat: runs, extraRuns: rules.runsPerNoBall, physicalRuns: runs,
                                    bowlerChargedRuns: rules.runsPerNoBall + runs,
                                    isLegal: rules.noBallIsLegalBall, faced: true)
        case .bye:
            return ResolvedDelivery(runsOffBat: 0, extraRuns: runs, physicalRuns: runs,
                                    bowlerChargedRuns: 0, isLegal: true, faced: true)
        case .legBye:
            return ResolvedDelivery(runsOffBat: 0, extraRuns: runs, physicalRuns: runs,
                                    bowlerChargedRuns: 0, isLegal: true, faced: true)
        }
    }

    // MARK: - State replay

    static func computeState(battingOrder: [UUID],
                             deliveries: [DeliveryData],
                             rules: InningsRules,
                             openers: (striker: UUID, nonStriker: UUID?)? = nil) -> InningsState {
        var state = InningsState()
        guard !battingOrder.isEmpty else { return state }

        if let openers {
            state.strikerID = openers.striker
            state.nonStrikerID = openers.nonStriker
        } else {
            state.strikerID = battingOrder.first
            state.nonStrikerID = battingOrder.count > 1 ? battingOrder[1] : nil
        }

        for d in deliveries.sorted(by: { $0.sequence < $1.sequence }) {
            state.justCompletedOver = false
            state.totalRuns += d.runsOffBat + d.extraRuns
            if d.isWicket { state.wickets += 1 }

            var overCompleted = false
            if d.isLegalDelivery {
                state.legalBalls += 1
                state.ballsThisOver += 1
                if state.ballsThisOver >= rules.ballsPerOver { overCompleted = true }
            }

            if let sa = d.strikerAfterID {
                // Explicit post-state (wickets, run-outs, manual corrections).
                state.strikerID = sa
                state.nonStrikerID = d.nonStrikerAfterID
            } else {
                if d.physicalRuns % 2 == 1 { swapStrike(&state) }
                if overCompleted { swapStrike(&state) }
            }

            if overCompleted {
                state.ballsThisOver = 0
                state.oversCompleted += 1
                state.justCompletedOver = true
            }

            state.isAllOut = state.wickets >= max(0, rules.lineupSize - 1)
            if let target = rules.target, state.totalRuns >= target { state.targetReached = true }
            state.isInningsComplete = state.isAllOut
                || state.oversCompleted >= rules.oversLimit
                || state.targetReached
            if state.isInningsComplete { break }
        }

        return state
    }

    private static func swapStrike(_ state: inout InningsState) {
        let s = state.strikerID
        state.strikerID = state.nonStrikerID
        state.nonStrikerID = s
    }

    /// Batters who have appeared at the crease (openers + everyone brought in on a wicket).
    static func appearedBatters(battingOrder: [UUID], deliveries: [DeliveryData],
                                openers: (striker: UUID, nonStriker: UUID?)? = nil) -> Set<UUID> {
        var seen = Set<UUID>()
        if let openers {
            seen.insert(openers.striker)
            if let n = openers.nonStriker { seen.insert(n) }
        } else {
            if let a = battingOrder.first { seen.insert(a) }
            if battingOrder.count > 1 { seen.insert(battingOrder[1]) }
        }
        for d in deliveries where d.newBatterID != nil {
            seen.insert(d.newBatterID!)
        }
        return seen
    }

    // MARK: - Commentary & highlights

    static func commentary(bowler: String, striker: String,
                           input: BallInput, resolved: ResolvedDelivery,
                           dismissedName: String?, fielderName: String?) -> String {
        let prefix = "\(bowler) to \(striker),"
        if input.isWicket {
            let how = input.dismissal?.displayName ?? "Out"
            var line = "\(prefix) OUT! \(how)"
            if let f = fielderName, let d = input.dismissal, d.involvesFielder {
                line += " (\(f))"
            }
            if let name = dismissedName { line += " — \(name) departs." } else { line += "." }
            return line
        }
        let runs = input.padRuns
        switch input.extra {
        case .wide:   return "\(prefix) Wide" + (runs > 0 ? " + \(runs)." : ".")
        case .noBall: return "\(prefix) No Ball" + (runs > 0 ? ", \(runs) off the bat." : ".")
        case .bye:    return "\(prefix) \(runs) bye\(runs == 1 ? "" : "s")."
        case .legBye: return "\(prefix) \(runs) leg bye\(runs == 1 ? "" : "s")."
        case .none:
            switch runs {
            case 0: return "\(prefix) no run."
            case 4: return "\(prefix) FOUR!"
            case 6: return "\(prefix) SIX!"
            case 1: return "\(prefix) 1 run."
            default: return "\(prefix) \(runs) runs."
            }
        }
    }

    static func highlightTags(for input: BallInput) -> [HighlightTag] {
        var tags: [HighlightTag] = []
        if input.isWicket { tags.append(.wicket) }
        if input.extra == .none {
            if input.padRuns == 4 { tags.append(.four) }
            if input.padRuns == 6 { tags.append(.six) }
        }
        return tags
    }
}
