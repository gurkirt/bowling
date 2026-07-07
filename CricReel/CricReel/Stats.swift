//
//  Stats.swift
//  CricReel
//
//  Derives per-player batting and bowling figures by aggregating deliveries.
//  All figures are computed on demand — nothing is denormalised.
//

import Foundation

struct BattingLine: Identifiable {
    var playerID: UUID
    var runs: Int = 0
    var ballsFaced: Int = 0
    var fours: Int = 0
    var sixes: Int = 0
    var isOut: Bool = false
    var dismissal: DismissalType? = nil

    var id: UUID { playerID }
    var strikeRate: Double {
        ballsFaced == 0 ? 0 : Double(runs) / Double(ballsFaced) * 100.0
    }
}

struct BowlingLine: Identifiable {
    var playerID: UUID
    var legalBalls: Int = 0
    var runsConceded: Int = 0
    var wickets: Int = 0

    var id: UUID { playerID }

    var oversDisplay: String {
        "\(legalBalls / 6).\(legalBalls % 6)"
    }
    var economy: Double {
        legalBalls == 0 ? 0 : Double(runsConceded) / (Double(legalBalls) / 6.0)
    }
}

enum StatsBuilder {

    /// Batting figures for every batter that faced a ball in the given deliveries.
    static func batting(from deliveries: [DeliveryData]) -> [UUID: BattingLine] {
        var lines: [UUID: BattingLine] = [:]

        func line(_ id: UUID) -> BattingLine { lines[id] ?? BattingLine(playerID: id) }

        for d in deliveries.sorted(by: { $0.sequence < $1.sequence }) {
            // Runs + boundaries are credited to the striker on legal deliveries.
            if d.extraType == .none {
                var l = line(d.strikerID)
                l.runs += d.runsOffBat
                if d.isLegalDelivery { l.ballsFaced += 1 }
                if d.runsOffBat == 4 { l.fours += 1 }
                if d.runsOffBat == 6 { l.sixes += 1 }
                lines[d.strikerID] = l
            }
            // Dismissal.
            if d.isWicket {
                let outID = d.dismissedPlayerID ?? d.strikerID
                var l = line(outID)
                l.isOut = true
                l.dismissal = d.dismissalType
                lines[outID] = l
            }
        }
        return lines
    }

    /// Bowling figures for every bowler in the given deliveries.
    static func bowling(from deliveries: [DeliveryData]) -> [UUID: BowlingLine] {
        var lines: [UUID: BowlingLine] = [:]
        func line(_ id: UUID) -> BowlingLine { lines[id] ?? BowlingLine(playerID: id) }

        for d in deliveries {
            var l = line(d.bowlerID)
            if d.isLegalDelivery { l.legalBalls += 1 }
            // v1: the only extra is a wide, which is charged to the bowler.
            l.runsConceded += d.runsOffBat + d.extraRuns
            if d.isWicket, let type = d.dismissalType, type.creditedToBowler {
                l.wickets += 1
            }
            lines[d.bowlerID] = l
        }
        return lines
    }
}
