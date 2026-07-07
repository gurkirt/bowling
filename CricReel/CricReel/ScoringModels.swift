//
//  ScoringModels.swift
//  CricReel
//
//  SwiftData model layer for local cricket scoring.
//  v1 scope: runs 0–6, wickets, and wides. Extras/dismissals enums are kept
//  extensible so byes/no-balls can be added later without a migration rewrite.
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

/// Delivery extra type. v1 supports `.none` and `.wide`; more can be added later.
enum ExtraType: String, Codable, CaseIterable {
    case none
    case wide

    var displayName: String {
        switch self {
        case .none: return "—"
        case .wide: return "Wide"
        }
    }
}

enum DismissalType: String, Codable, CaseIterable {
    case bowled
    case caught
    case lbw
    case runOut
    case stumped
    case hitWicket
    case other

    var displayName: String {
        switch self {
        case .bowled:    return "Bowled"
        case .caught:    return "Caught"
        case .lbw:       return "LBW"
        case .runOut:    return "Run Out"
        case .stumped:   return "Stumped"
        case .hitWicket: return "Hit Wicket"
        case .other:     return "Other"
        }
    }

    /// Whether the dismissal is credited to the bowler.
    var creditedToBowler: Bool {
        switch self {
        case .bowled, .caught, .lbw, .stumped, .hitWicket: return true
        case .runOut, .other: return false
        }
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
    var createdAt: Date
    var battingStyle: String?
    var bowlingStyle: String?

    init(id: UUID = UUID(),
         name: String,
         battingStyle: String? = nil,
         bowlingStyle: String? = nil,
         createdAt: Date = .now) {
        self.id = id
        self.name = name
        self.battingStyle = battingStyle
        self.bowlingStyle = bowlingStyle
        self.createdAt = createdAt
    }
}

// MARK: - Team

@Model
final class Team {
    @Attribute(.unique) var id: UUID
    var name: String
    var createdAt: Date
    /// Squad pool. Order is not significant here; batting order lives on the Match.
    @Relationship var players: [Player]

    init(id: UUID = UUID(),
         name: String,
         players: [Player] = [],
         createdAt: Date = .now) {
        self.id = id
        self.name = name
        self.players = players
        self.createdAt = createdAt
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
    var runsPerWide: Int
    /// Whether a wide counts as a legal delivery (advances the over). Standard cricket: false.
    var wideIsLegalBall: Bool

    // Teams (references + name snapshots so a scorecard survives team edits)
    @Relationship var teamA: Team?
    @Relationship var teamB: Team?
    var teamAName: String
    var teamBName: String

    /// Playing XI as ordered player IDs (index = batting order). Kept as ID arrays
    /// because SwiftData to-many relationships are unordered.
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
         runsPerWide: Int = 1,
         wideIsLegalBall: Bool = false,
         teamA: Team? = nil,
         teamB: Team? = nil,
         teamAName: String = "",
         teamBName: String = "",
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
        self.runsPerWide = runsPerWide
        self.wideIsLegalBall = wideIsLegalBall
        self.teamA = teamA
        self.teamB = teamB
        self.teamAName = teamAName
        self.teamBName = teamBName
        self.teamALineupIDs = teamALineupIDs
        self.teamBLineupIDs = teamBLineupIDs
        self.tossWinnerIsA = tossWinnerIsA
        self.tossDecision = tossDecision
        self.innings = innings
    }

    /// Which team bats first, derived from toss winner + decision.
    var battingFirstIsA: Bool {
        // Winner bats if they chose bat; otherwise the other team bats.
        tossWinnerIsA ? (tossDecision == .bat) : (tossDecision == .bowl)
    }

    func lineupIDs(isTeamA: Bool) -> [UUID] {
        isTeamA ? teamALineupIDs : teamBLineupIDs
    }

    func teamName(isTeamA: Bool) -> String {
        isTeamA ? teamAName : teamBName
    }
}

// MARK: - Innings

@Model
final class Innings {
    @Attribute(.unique) var id: UUID
    var order: Int              // 1 or 2
    var battingTeamIsA: Bool
    var isComplete: Bool

    @Relationship(deleteRule: .cascade, inverse: \Delivery.innings) var deliveries: [Delivery]
    var match: Match?

    init(id: UUID = UUID(),
         order: Int,
         battingTeamIsA: Bool,
         isComplete: Bool = false,
         deliveries: [Delivery] = []) {
        self.id = id
        self.order = order
        self.battingTeamIsA = battingTeamIsA
        self.isComplete = isComplete
        self.deliveries = deliveries
    }

    /// Deliveries in scoring order.
    var orderedDeliveries: [Delivery] {
        deliveries.sorted { $0.sequence < $1.sequence }
    }
}

// MARK: - Delivery (one ball)

@Model
final class Delivery {
    @Attribute(.unique) var id: UUID
    var timestamp: Date
    /// Global order within the innings (monotonically increasing).
    var sequence: Int
    /// 0-based over index.
    var overNumber: Int
    /// 1-based legal-ball number within the over (for legal balls); for a wide this is
    /// the number the next legal ball will take.
    var ballInOver: Int

    var strikerID: UUID
    var nonStrikerID: UUID
    var bowlerID: UUID

    var runsOffBat: Int
    var extraType: ExtraType
    var extraRuns: Int

    var isWicket: Bool
    var dismissalType: DismissalType?
    var dismissedPlayerID: UUID?

    /// Whether this delivery advances the over count (false for a wide by default).
    var isLegalDelivery: Bool

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
         isWicket: Bool = false,
         dismissalType: DismissalType? = nil,
         dismissedPlayerID: UUID? = nil,
         isLegalDelivery: Bool = true,
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
        self.isWicket = isWicket
        self.dismissalType = dismissalType
        self.dismissedPlayerID = dismissedPlayerID
        self.isLegalDelivery = isLegalDelivery
        self.clipFilename = clipFilename
        self.commentary = commentary
        self.highlightTags = highlightTags
    }

    /// Total runs credited on this delivery (bat + extras).
    var totalRuns: Int { runsOffBat + extraRuns }
}
