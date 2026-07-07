//
//  MatchSetupView.swift
//  CricReel
//
//  Match creation: teams, format (overs, players-per-side, extras), playing XI with
//  squad validation, and toss.
//

import SwiftUI
import SwiftData

struct MatchSetupView: View {
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \Team.name) private var teams: [Team]

    var onCreated: (Match) -> Void

    @State private var teamA: Team?
    @State private var teamB: Team?
    @State private var venue = ""
    @State private var overs = 6
    @State private var playersPerSide = 11
    @State private var runsPerWide = 1
    @State private var runsPerNoBall = 1
    @State private var lineupA: [Player] = []
    @State private var lineupB: [Player] = []
    @State private var tossWinnerIsA = true
    @State private var tossDecision: TossDecision = .bat

    @FocusState private var venueFocused: Bool

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
                        .focused($venueFocused)
                        .submitLabel(.done)
                        .onSubmit { venueFocused = false }
                }

                Section("Format") {
                    Stepper("Overs per innings: \(overs)", value: $overs, in: 1...50)
                    Stepper("Players per side: \(playersPerSide)", value: $playersPerSide, in: 2...11)
                        .onChange(of: playersPerSide) { _, new in
                            if lineupA.count > new { lineupA = Array(lineupA.prefix(new)) }
                            if lineupB.count > new { lineupB = Array(lineupB.prefix(new)) }
                        }
                    Stepper("Runs per wide: \(runsPerWide)", value: $runsPerWide, in: 1...5)
                    Stepper("Runs per no-ball: \(runsPerNoBall)", value: $runsPerNoBall, in: 1...5)
                }

                if let a = teamA { lineupSection(team: a, lineup: $lineupA) }
                if let b = teamB { lineupSection(team: b, lineup: $lineupB) }

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
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("New Match")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) { Button("Create") { create() }.disabled(!canCreate) }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { venueFocused = false }
                }
            }
        }
    }

    @ViewBuilder
    private func lineupSection(team: Team, lineup: Binding<[Player]>) -> some View {
        Section("\(team.name) — Playing XI (\(lineup.wrappedValue.count)/\(playersPerSide))") {
            if team.players.count < playersPerSide {
                VStack(alignment: .leading, spacing: 6) {
                    Label("Squad has \(team.players.count) players, need \(playersPerSide).",
                          systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                    NavigationLink {
                        TeamEditView(team: team)
                    } label: {
                        Label("Add players to \(team.name)", systemImage: "person.badge.plus")
                    }
                }
            }
            NavigationLink {
                LineupPicker(team: team, required: playersPerSide, lineup: lineup)
            } label: {
                HStack {
                    Text("Select batting order")
                    Spacer()
                    Text(lineup.wrappedValue.count == playersPerSide ? "Ready" : "\(lineup.wrappedValue.count)/\(playersPerSide)")
                        .foregroundStyle(lineup.wrappedValue.count == playersPerSide ? .green : .secondary)
                }
            }
        }
    }

    private var canCreate: Bool {
        guard let a = teamA, let b = teamB, a.id != b.id else { return false }
        return lineupA.count == playersPerSide && lineupB.count == playersPerSide
    }

    private func create() {
        guard let a = teamA, let b = teamB else { return }
        let match = Match(
            venue: venue,
            status: .live,
            oversPerInnings: overs,
            playersPerSide: playersPerSide,
            runsPerWide: runsPerWide,
            runsPerNoBall: runsPerNoBall,
            teamA: a, teamB: b,
            teamAName: a.name, teamBName: b.name,
            teamALineupIDs: lineupA.map(\.id),
            teamBLineupIDs: lineupB.map(\.id),
            tossWinnerIsA: tossWinnerIsA,
            tossDecision: tossDecision)
        let first = Innings(order: 1, battingTeamIsA: match.battingFirstIsA)
        context.insert(match)
        first.match = match
        onCreated(match)
        dismiss()
    }
}

/// Select and order a team's playing XI (selection order = batting order).
struct LineupPicker: View {
    let team: Team
    let required: Int
    @Binding var lineup: [Player]

    var body: some View {
        List {
            Section("Batting order (\(lineup.count)/\(required))") {
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
                        if lineup.count < required { lineup.append(player) }
                    } label: {
                        HStack {
                            Text(player.name).foregroundStyle(lineup.count < required ? .primary : .secondary)
                            Spacer()
                            Image(systemName: "plus.circle").foregroundStyle(.blue)
                        }
                    }
                    .disabled(lineup.count >= required)
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
