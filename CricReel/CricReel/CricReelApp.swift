//
//  CricReelApp.swift
//  CricReel
//

import SwiftUI
import SwiftData

@main
struct CricReelApp: App {
    let container: ModelContainer

    init() {
        let schema = Schema([Player.self, Team.self, Match.self, Innings.self, Delivery.self])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            container = try ModelContainer(for: schema, configurations: config)
        } catch {
            // The on-disk store is from an incompatible earlier schema and cannot be
            // migrated automatically. In this local/dev phase we reset it rather than crash.
            print("⚠️ [CricReel] Store incompatible (\(error)). Resetting local database.")
            Self.deleteStore(at: config.url)
            do {
                container = try ModelContainer(for: schema, configurations: config)
            } catch {
                fatalError("Failed to create a fresh ModelContainer: \(error)")
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            AppRootView()
        }
        .modelContainer(container)
    }

    /// Remove the SwiftData store file and its -wal/-shm siblings.
    private static func deleteStore(at url: URL) {
        let fm = FileManager.default
        let dir = url.deletingLastPathComponent()
        let base = url.lastPathComponent
        for name in [base, base + "-wal", base + "-shm"] {
            try? fm.removeItem(at: dir.appendingPathComponent(name))
        }
    }
}
