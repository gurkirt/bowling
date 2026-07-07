//
//  DeliveryFormatting.swift
//  CricReel
//
//  Shared presentation helpers for a delivery: short badge, outcome text, colour,
//  and the two-line overlay burned onto clips / shown during replay.
//

import SwiftUI

enum DeliveryKind {
    case four, six, wicket, dot, extra, normal

    var color: Color {
        switch self {
        case .four:   return .green
        case .six:    return .purple
        case .wicket: return .red
        case .dot:    return Color(.systemGray3)
        case .extra:  return .orange
        case .normal: return Color(.systemGray)
        }
    }
}

enum DeliveryFormatting {

    static func kind(_ d: Delivery) -> DeliveryKind {
        if d.isWicket { return .wicket }
        if d.extraType == .none {
            if d.runsOffBat == 4 { return .four }
            if d.runsOffBat == 6 { return .six }
            if d.runsOffBat == 0 { return .dot }
            return .normal
        }
        return .extra
    }

    /// Short badge for the this-over strip and ball-by-ball list ("4", "6", "W", "1", "•", "Wd").
    static func badge(_ d: Delivery) -> String {
        if d.isWicket { return "W" }
        switch d.extraType {
        case .none:   return d.runsOffBat == 0 ? "•" : "\(d.runsOffBat)"
        case .wide:   return d.physicalRuns > 0 ? "Wd\(d.physicalRuns)" : "Wd"
        case .noBall: return d.runsOffBat > 0 ? "Nb\(d.runsOffBat)" : "Nb"
        case .bye:    return "\(d.extraRuns)b"
        case .legBye: return "\(d.extraRuns)lb"
        }
    }

    /// Outcome phrase used in commentary and clip overlays.
    static func outcome(_ d: Delivery) -> String {
        if d.isWicket {
            return "OUT" + (d.dismissalType.map { " · \($0.displayName)" } ?? "")
        }
        switch d.extraType {
        case .wide:   return "Wide" + (d.physicalRuns > 0 ? " +\(d.physicalRuns)" : "")
        case .noBall: return "No Ball" + (d.runsOffBat > 0 ? ", \(d.runsOffBat)" : "")
        case .bye:    return "\(d.extraRuns) Bye\(d.extraRuns == 1 ? "" : "s")"
        case .legBye: return "\(d.extraRuns) Leg Bye\(d.extraRuns == 1 ? "" : "s")"
        case .none:
            switch d.runsOffBat {
            case 0: return "No run"
            case 4: return "FOUR"
            case 6: return "SIX"
            case 1: return "1 run"
            default: return "\(d.runsOffBat) runs"
            }
        }
    }

    /// "Bowler to Striker, FOUR"
    static func description(_ d: Delivery, lookup: PlayerLookup) -> String {
        "\(lookup.name(d.bowlerID)) to \(lookup.name(d.strikerID)), \(outcome(d))"
    }

    /// Two-line overlay: ("2.3 Bowler → Batter", "FOUR").
    static func overlayLines(_ d: Delivery, lookup: PlayerLookup) -> (String, String) {
        ("\(d.overNumber).\(d.ballInOver) \(lookup.name(d.bowlerID)) → \(lookup.name(d.strikerID))",
         outcome(d))
    }
}
