//
//  ScoringModels.swift
//  CricReel
//
//  SwiftData model layer for local cricket scoring.
//  Supports runs 0–6, wickets, and extras (wide, no-ball, bye, leg-bye).
//

import Foundation
import SwiftData

// MARK: - Enums

enum MatchStatus: String, Codable, CaseIterable {
    case setup
    case live
    case completed
}

enum TossDecision: String, Codable, CaseIterable {
    case bat
    case bowl
}

/// Delivery extra type.
enum ExtraType: String, Codable, CaseIterable, Identifiable {
    case none
    case wide
    case noBall
    case bye
    case legBye

    var id: String { rawValue }

    var shortLabel: String {
        switch self {
        case .none:   return ""
        case .wide:   return "WD"
        case .noBall: return "NB"
        case .bye:    return "B"
        case .legBye: return "LB"
        }
    }

    var displayName: String {
        switch self {
        case .none:   return "—"
        case .wide:   return "Wide"
        case .noBall: return "No Ball"
        case .bye:    return "Bye"
        case .legBye: return "Leg Bye"
        }
    }
}

enum DismissalType: String, Codable, CaseIterable, Identifiable {
    case bowled
    case caught
    case caughtAndBowled
    case lbw
    case runOut
    case stumped
    case hitWicket
    case other

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .bowled:          return "Bowled"
        case .caught:          return "Caught"
        case .caughtAndBowled: return "C&B"
        case .lbw:             return "LBW"
        case .runOut:          return "Run Out"
        case .stumped:         return "Stumped"
        case .hitWicket:       return "Hit Wicket"
        case .other:           return "Other"
        }
    }

    /// Whether the dismissal is credited to the bowler.
    var creditedToBowler: Bool {
        switch self {
        case .bowled, .caught, .caughtAndBowled, .lbw, .stumped, .hitWicket: return true
        case .runOut, .other: return false
        }
    }

    /// Whether a separate fielder is involved (C&B is the bowler, so no).
    var involvesFielder: Bool {
        self == .caught || self == .runOut || self == .stumped
    }

    /// Whether the scorer must be asked who ends up on strike (only run-out / other).
    /// For every other dismissal the incoming batter takes the striker's end.
    var asksStrike: Bool {
        self == .runOut || self == .other
    }
}

enum HighlightTag: String, Codable, CaseIterable, Identifiable {
    case four
    case six
    case wicket

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .four:   return "Fours"
        case .six:    return "Sixes"
        case .wicket: return "Wickets"
        }
    }
}

// MARK: - Player

@Model
final class Player {
    @Attribute(.unique) var id: UUID
    var name: String
    /// Short display name (≤12 chars) used on the scoreboard and reels.
    var reelName: String = ""
    var createdAt: Date
    var battingStyle: String?
    var bowlingStyle: String?

    init(id: UUID = UUID(),
         name: String,
         reelName: String = "",
         battingStyle: String? = nil,
         bowlingStyle: String? = nil,
         createdAt: Date = .now) {
        self.id = id
        self.name = name
        self.reelName = reelName
        self.battingStyle = battingStyle
        self.bowlingStyle = bowlingStyle
        self.createdAt = createdAt
    }

    /// Reel name if set, else a sensible short fallback from the full name.
    var effectiveReelName: String {
        let trimmed = reelName.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { return String(trimmed.prefix(12)) }
        return String(name.prefix(12))
    }
}

// MARK: - Team

@Model
final class Team {
    @Attribute(.unique) var id: UUID
    var name: String
    /// Short display name (≤6 chars) used on the scoreboard and reels.
    var reelName: String = ""
    var createdAt: Date
    @Relationship var players: [Player]

    init(id: UUID = UUID(),
         name: String,
         reelName: String = "",
         players: [Player] = [],
         createdAt: Date = .now) {
        self.id = id
        self.name = name
        self.reelName = reelName
        self.players = players
        self.createdAt = createdAt
    }

    var effectiveReelName: String {
        let trimmed = reelName.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { return String(trimmed.prefix(6)) }
        return String(name.prefix(6))
    }
}

// MARK: - Match

@Model
final class Match {
    @Attribute(.unique) var id: UUID
    var date: Date
    var venue: String
    var status: MatchStatus

    // Config
    var oversPerInnings: Int
    var ballsPerOver: Int
    var playersPerSide: Int
    var runsPerWide: Int
    var runsPerNoBall: Int
    var wideIsLegalBall: Bool
    var noBallIsLegalBall: Bool
    /// Max overs a single bowler may bowl (0 = unlimited).
    var maxOversPerBowler: Int = 0
    /// Minimum number of distinct bowlers expected (informational / validation).
    var minBowlers: Int = 2

    // Teams
    @Relationship var teamA: Team?
    @Relationship var teamB: Team?
    var teamAName: String
    var teamBName: String
    var teamAReelName: String = ""
    var teamBReelName: String = ""

    /// Playing XI as ordered player IDs (index = batting order).
    var teamALineupIDs: [UUID]
    var teamBLineupIDs: [UUID]

    // Toss
    var tossWinnerIsA: Bool
    var tossDecision: TossDecision

    @Relationship(deleteRule: .cascade, inverse: \Innings.match) var innings: [Innings]

    init(id: UUID = UUID(),
         date: Date = .now,
         venue: String = "",
         status: MatchStatus = .setup,
         oversPerInnings: Int = 6,
         ballsPerOver: Int = 6,
         playersPerSide: Int = 11,
         runsPerWide: Int = 1,
         runsPerNoBall: Int = 1,
         wideIsLegalBall: Bool = false,
         noBallIsLegalBall: Bool = false,
         maxOversPerBowler: Int = 0,
         minBowlers: Int = 2,
         teamA: Team? = nil,
         teamB: Team? = nil,
         teamAName: String = "",
         teamBName: String = "",
         teamAReelName: String = "",
         teamBReelName: String = "",
         teamALineupIDs: [UUID] = [],
         teamBLineupIDs: [UUID] = [],
         tossWinnerIsA: Bool = true,
         tossDecision: TossDecision = .bat,
         innings: [Innings] = []) {
        self.id = id
        self.date = date
        self.venue = venue
        self.status = status
        self.oversPerInnings = oversPerInnings
        self.ballsPerOver = ballsPerOver
        self.playersPerSide = playersPerSide
        self.runsPerWide = runsPerWide
        self.runsPerNoBall = runsPerNoBall
        self.wideIsLegalBall = wideIsLegalBall
        self.noBallIsLegalBall = noBallIsLegalBall
        self.maxOversPerBowler = maxOversPerBowler
        self.minBowlers = minBowlers
        self.teamA = teamA
        self.teamB = teamB
        self.teamAName = teamAName
        self.teamBName = teamBName
        self.teamAReelName = teamAReelName
        self.teamBReelName = teamBReelName
        self.teamALineupIDs = teamALineupIDs
        self.teamBLineupIDs = teamBLineupIDs
        self.tossWinnerIsA = tossWinnerIsA
        self.tossDecision = tossDecision
        self.innings = innings
    }

    var battingFirstIsA: Bool {
        tossWinnerIsA ? (tossDecision == .bat) : (tossDecision == .bowl)
    }

    func lineupIDs(isTeamA: Bool) -> [UUID] {
        isTeamA ? teamALineupIDs : teamBLineupIDs
    }

    func teamName(isTeamA: Bool) -> String {
        isTeamA ? teamAName : teamBName
    }

    func teamReelName(isTeamA: Bool) -> String {
        let raw = (isTeamA ? teamAReelName : teamBReelName).trimmingCharacters(in: .whitespaces)
        if !raw.isEmpty { return String(raw.prefix(6)) }
        return String(teamName(isTeamA: isTeamA).prefix(6))
    }
}

// MARK: - Innings

@Model
final class Innings {
    @Attribute(.unique) var id: UUID
    var order: Int              // 1 or 2
    var battingTeamIsA: Bool
    var isComplete: Bool
    /// Chosen opening batters (nil until the scorer picks them at the innings start).
    var openerStrikerID: UUID?
    var openerNonStrikerID: UUID?

    @Relationship(deleteRule: .cascade, inverse: \Delivery.innings) var deliveries: [Delivery]
    var match: Match?

    init(id: UUID = UUID(),
         order: Int,
         battingTeamIsA: Bool,
         isComplete: Bool = false,
         openerStrikerID: UUID? = nil,
         openerNonStrikerID: UUID? = nil,
         deliveries: [Delivery] = []) {
        self.id = id
        self.order = order
        self.battingTeamIsA = battingTeamIsA
        self.isComplete = isComplete
        self.openerStrikerID = openerStrikerID
        self.openerNonStrikerID = openerNonStrikerID
        self.deliveries = deliveries
    }

    var orderedDeliveries: [Delivery] {
        deliveries.sorted { $0.sequence < $1.sequence }
    }
}

// MARK: - Delivery (one ball)

@Model
final class Delivery {
    @Attribute(.unique) var id: UUID
    var timestamp: Date
    var sequence: Int
    var overNumber: Int
    var ballInOver: Int

    // Who was on the field for this ball.
    var strikerID: UUID
    var nonStrikerID: UUID
    var bowlerID: UUID

    // Runs
    var runsOffBat: Int
    var extraType: ExtraType
    var extraRuns: Int
    /// Number of runs physically completed by the batters (drives strike rotation).
    var physicalRuns: Int
    /// Runs charged to the bowler (off bat + wide/no-ball penalties + off-noball runs; excludes byes/leg-byes).
    var bowlerChargedRuns: Int

    // Ball accounting
    var isLegalDelivery: Bool     // advances the over
    var facedByBatsman: Bool      // counts toward striker's balls faced

    // Wicket
    var isWicket: Bool
    var dismissalType: DismissalType?
    var dismissedPlayerID: UUID?
    var fielderID: UUID?
    var newBatterID: UUID?

    /// Explicit post-delivery batters (set for wickets and manual strike corrections).
    /// When present, the engine uses these instead of computing rotation.
    var strikerAfterID: UUID?
    var nonStrikerAfterID: UUID?

    var clipFilename: String?
    var commentary: String
    var highlightTags: [HighlightTag]

    var innings: Innings?

    init(id: UUID = UUID(),
         timestamp: Date = .now,
         sequence: Int,
         overNumber: Int,
         ballInOver: Int,
         strikerID: UUID,
         nonStrikerID: UUID,
         bowlerID: UUID,
         runsOffBat: Int = 0,
         extraType: ExtraType = .none,
         extraRuns: Int = 0,
         physicalRuns: Int = 0,
         bowlerChargedRuns: Int = 0,
         isLegalDelivery: Bool = true,
         facedByBatsman: Bool = true,
         isWicket: Bool = false,
         dismissalType: DismissalType? = nil,
         dismissedPlayerID: UUID? = nil,
         fielderID: UUID? = nil,
         newBatterID: UUID? = nil,
         strikerAfterID: UUID? = nil,
         nonStrikerAfterID: UUID? = nil,
         clipFilename: String? = nil,
         commentary: String = "",
         highlightTags: [HighlightTag] = []) {
        self.id = id
        self.timestamp = timestamp
        self.sequence = sequence
        self.overNumber = overNumber
        self.ballInOver = ballInOver
        self.strikerID = strikerID
        self.nonStrikerID = nonStrikerID
        self.bowlerID = bowlerID
        self.runsOffBat = runsOffBat
        self.extraType = extraType
        self.extraRuns = extraRuns
        self.physicalRuns = physicalRuns
        self.bowlerChargedRuns = bowlerChargedRuns
        self.isLegalDelivery = isLegalDelivery
        self.facedByBatsman = facedByBatsman
        self.isWicket = isWicket
        self.dismissalType = dismissalType
        self.dismissedPlayerID = dismissedPlayerID
        self.fielderID = fielderID
        self.newBatterID = newBatterID
        self.strikerAfterID = strikerAfterID
        self.nonStrikerAfterID = nonStrikerAfterID
        self.clipFilename = clipFilename
        self.commentary = commentary
        self.highlightTags = highlightTags
    }

    var totalRuns: Int { runsOffBat + extraRuns }
}
