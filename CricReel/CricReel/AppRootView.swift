//
//  AppRootView.swift
//  CricReel
//
//  Top-level tab navigation.
//

import SwiftUI

struct AppRootView: View {
    @ObservedObject private var settings = SettingsStore.shared

    var body: some View {
        TabView {
            MatchListView()
                .tabItem { Label("Matches", systemImage: "figure.cricket") }
            PlayerStatsView()
                .tabItem { Label("Journeys", systemImage: "chart.bar") }
            PlayerPoolView()
                .tabItem { Label("Players", systemImage: "person.3") }
            TeamListView()
                .tabItem { Label("Teams", systemImage: "person.3.sequence") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
        .preferredColorScheme(settings.appearance.colorScheme)
    }
}
