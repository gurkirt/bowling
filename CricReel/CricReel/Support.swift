//
//  Support.swift
//  CricReel
//
//  Small shared helpers: file locations for clips/reels and player-name lookup.
//

import Foundation
import SwiftData

extension Notification.Name {
    /// Posted by `VideoWriter` when a triggered clip finishes writing. Object is the file URL.
    static let newClipSaved = Notification.Name("CricReelNewClipSaved")
}

enum ClipStore {
    static var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Full URL for a clip filename stored in the Documents directory.
    static func url(forClip filename: String) -> URL {
        documentsDirectory.appendingPathComponent(filename)
    }

    static func clipExists(_ filename: String?) -> Bool {
        guard let filename else { return false }
        return FileManager.default.fileExists(atPath: url(forClip: filename).path)
    }

    /// Directory where stitched highlight reels are written.
    static var highlightsDirectory: URL {
        let dir = documentsDirectory.appendingPathComponent("Highlights", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }
}

/// Resolves player names from IDs. Build once per view from a `@Query` of players.
struct PlayerLookup {
    private let namesByID: [UUID: String]
    private let reelByID: [UUID: String]

    init(_ players: [Player]) {
        namesByID = Dictionary(players.map { ($0.id, $0.name) }, uniquingKeysWith: { a, _ in a })
        reelByID = Dictionary(players.map { ($0.id, $0.effectiveReelName) }, uniquingKeysWith: { a, _ in a })
    }

    func name(_ id: UUID?) -> String {
        guard let id else { return "—" }
        return namesByID[id] ?? "Unknown"
    }

    /// Short display name for scoreboard / reels.
    func reel(_ id: UUID?) -> String {
        guard let id else { return "—" }
        return reelByID[id] ?? namesByID[id] ?? "?"
    }
}
