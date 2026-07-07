//
//  MatchSetupView.swift
//  CricReel
//
//  Multi-step match creation: teams, config, playing XI + batting order, toss.
//

import SwiftUI
import SwiftData

struct MatchSetupView: View {
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \Team.name) private var teams: [Team]

    /// Called with the created match so the caller can navigate into it.
    var onCreated: (Match) -> Void

    @State private var teamA: Team?
    @State private var teamB: Team?
    @State private var venue = ""
    @State private var overs = 6
    @State private var runsPerWide = 1
    @State private var lineupA: [Player] = []
    @State private var lineupB: [Player] = []
    @State private var tossWinnerIsA = true
    @State private var tossDecision: TossDecision = .bat

    var body: some View {
        NavigationStack {
            Form {
                Section("Teams") {
                    Picker("Team A", selection: $teamA) {
                        Text("Select").tag(Team?.none)
                        ForEach(teams) { Text($0.name).tag(Team?.some($0)) }
                    }
                    Picker("Team B", selection: $teamB) {
                        Text("Select").tag(Team?.none)
                        ForEach(teams) { Text($0.name).tag(Team?.some($0)) }
                    }
                    if let a = teamA, let b = teamB, a.id == b.id {
                        Text("Team A and Team B must be different.")
                            .font(.caption).foregroundStyle(.red)
                    }
                    TextField("Venue (optional)", text: $venue)
                }

                Section("Format") {
                    Stepper("Overs per innings: \(overs)", value: $overs, in: 1...50)
                    Stepper("Runs per wide: \(runsPerWide)", value: $runsPerWide, in: 1...5)
                }

                if let a = teamA {
                    Section("\(a.name) — Playing XI (batting order)") {
                        NavigationLink {
                            LineupPicker(team: a, lineup: $lineupA)
                        } label: {
                            lineupSummary(lineupA)
                        }
                    }
                }
                if let b = teamB {
                    Section("\(b.name) — Playing XI (batting order)") {
                        NavigationLink {
                            LineupPicker(team: b, lineup: $lineupB)
                        } label: {
                            lineupSummary(lineupB)
                        }
                    }
                }

                Section("Toss") {
                    Picker("Won by", selection: $tossWinnerIsA) {
                        Text(teamA?.name ?? "Team A").tag(true)
                        Text(teamB?.name ?? "Team B").tag(false)
                    }
                    Picker("Elected to", selection: $tossDecision) {
                        Text("Bat").tag(TossDecision.bat)
                        Text("Bowl").tag(TossDecision.bowl)
                    }
                }
            }
            .navigationTitle("New Match")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") { create() }.disabled(!canCreate)
                }
            }
        }
    }

    private func lineupSummary(_ lineup: [Player]) -> some View {
        HStack {
            Text("Playing XI")
            Spacer()
            Text(lineup.isEmpty ? "None" : "\(lineup.count) selected")
                .foregroundStyle(.secondary)
        }
    }

    private var canCreate: Bool {
        guard let a = teamA, let b = teamB, a.id != b.id else { return false }
        return lineupA.count >= 2 && lineupB.count >= 2
    }

    private func create() {
        guard let a = teamA, let b = teamB else { return }
        let match = Match(
            venue: venue,
            status: .live,
            oversPerInnings: overs,
            runsPerWide: runsPerWide,
            teamA: a,
            teamB: b,
            teamAName: a.name,
            teamBName: b.name,
            teamALineupIDs: lineupA.map(\.id),
            teamBLineupIDs: lineupB.map(\.id),
            tossWinnerIsA: tossWinnerIsA,
            tossDecision: tossDecision)

        // Create the first innings based on the toss.
        let firstInnings = Innings(order: 1, battingTeamIsA: match.battingFirstIsA)
        context.insert(match)
        firstInnings.match = match
        onCreated(match)
        dismiss()
    }
}

/// Select and order a team's playing XI. Selection order defines the batting order,
/// and rows can be reordered with drag handles.
struct LineupPicker: View {
    let team: Team
    @Binding var lineup: [Player]

    var body: some View {
        List {
            Section("Batting order") {
                if lineup.isEmpty {
                    Text("Tap players below to add them.").foregroundStyle(.secondary)
                }
                ForEach(Array(lineup.enumerated()), id: \.element.id) { index, player in
                    HStack {
                        Text("\(index + 1).").foregroundStyle(.secondary).monospacedDigit()
                        Text(player.name)
                    }
                }
                .onMove { lineup.move(fromOffsets: $0, toOffset: $1) }
                .onDelete { lineup.remove(atOffsets: $0) }
            }
            Section("Available") {
                ForEach(available) { player in
                    Button {
                        lineup.append(player)
                    } label: {
                        HStack {
                            Text(player.name).foregroundStyle(.primary)
                            Spacer()
                            Image(systemName: "plus.circle").foregroundStyle(.blue)
                        }
                    }
                }
            }
        }
        .environment(\.editMode, .constant(.active))
        .navigationTitle("\(team.name) XI")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var available: [Player] {
        team.players
            .filter { p in !lineup.contains { $0.id == p.id } }
            .sorted { $0.name < $1.name }
    }
}
