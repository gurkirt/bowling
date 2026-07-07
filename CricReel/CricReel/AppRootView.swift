//
//  AppRootView.swift
//  CricReel
//
//  Top-level tab navigation.
//

import SwiftUI

struct AppRootView: View {
    var body: some View {
        TabView {
            MatchListView()
                .tabItem { Label("Matches", systemImage: "sportscourt") }
            PlayerStatsView()
                .tabItem { Label("Journeys", systemImage: "chart.bar") }
            PlayerPoolView()
                .tabItem { Label("Players", systemImage: "person.3") }
            TeamListView()
                .tabItem { Label("Teams", systemImage: "person.3.sequence") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
    }
}
