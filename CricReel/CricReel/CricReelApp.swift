//
//  CricReelApp.swift
//  CricReel
//

import SwiftUI
import SwiftData

@main
struct CricReelApp: App {
    var body: some Scene {
        WindowGroup {
            AppRootView()
        }
        .modelContainer(for: [Player.self, Team.self, Match.self, Innings.self, Delivery.self])
    }
}
