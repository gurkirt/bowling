//
//  TeamSetupView.swift
//  CricReel
//
//  Create/edit teams and pick their squad from the player pool.
//

import SwiftUI
import SwiftData

struct TeamListView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \Team.name) private var teams: [Team]

    @State private var showingNew = false

    var body: some View {
        NavigationStack {
            List {
                if teams.isEmpty {
                    ContentUnavailableView("No Teams Yet",
                                           systemImage: "person.3.sequence",
                                           description: Text("Create a team and add players to its squad."))
                }
                ForEach(teams) { team in
                    NavigationLink {
                        TeamEditView(team: team)
                    } label: {
                        VStack(alignment: .leading) {
                            Text(team.name).font(.headline)
                            Text("\(team.players.count) players · reel \(team.effectiveReelName)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { offsets in
                    for i in offsets { context.delete(teams[i]) }
                }
            }
            .navigationTitle("Teams")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showingNew = true } label: { Image(systemName: "plus") }
                }
            }
            .sheet(isPresented: $showingNew) { NewTeamSheet() }
        }
    }
}

struct NewTeamSheet: View {
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var reel = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Team") {
                    TextField("Team name", text: $name)
                    TextField("Reel name (≤6, shown on scoreboard)", text: $reel)
                        .onChange(of: reel) { _, v in if v.count > 6 { reel = String(v.prefix(6)) } }
                }
            }
            .navigationTitle("New Team")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") {
                        let n = name.trimmingCharacters(in: .whitespaces)
                        if !n.isEmpty {
                            context.insert(Team(name: n, reelName: reel.trimmingCharacters(in: .whitespaces)))
                        }
                        dismiss()
                    }
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
    }
}

struct TeamEditView: View {
    @Bindable var team: Team
    @Query(sort: \Player.name) private var allPlayers: [Player]

    var body: some View {
        Form {
            Section("Name") {
                TextField("Team name", text: $team.name)
                TextField("Reel name (≤6)", text: $team.reelName)
                    .onChange(of: team.reelName) { _, v in if v.count > 6 { team.reelName = String(v.prefix(6)) } }
            }
            Section("Squad") {
                if allPlayers.isEmpty {
                    Text("Add players in the Players tab first.")
                        .foregroundStyle(.secondary)
                }
                ForEach(allPlayers) { player in
                    Button {
                        toggle(player)
                    } label: {
                        HStack {
                            Text(player.name).foregroundStyle(.primary)
                            Spacer()
                            if isMember(player) {
                                Image(systemName: "checkmark.circle.fill").foregroundStyle(.blue)
                            } else {
                                Image(systemName: "circle").foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle(team.name.isEmpty ? "Team" : team.name)
        .navigationBarTitleDisplayMode(.inline)
    }

    private func isMember(_ player: Player) -> Bool {
        team.players.contains { $0.id == player.id }
    }

    private func toggle(_ player: Player) {
        if let idx = team.players.firstIndex(where: { $0.id == player.id }) {
            team.players.remove(at: idx)
        } else {
            team.players.append(player)
        }
    }
}
