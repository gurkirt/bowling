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
    @State private var newTeamName = ""

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
                            Text("\(team.players.count) players").font(.caption).foregroundStyle(.secondary)
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
            .alert("New Team", isPresented: $showingNew) {
                TextField("Team name", text: $newTeamName)
                Button("Create") {
                    let name = newTeamName.trimmingCharacters(in: .whitespaces)
                    if !name.isEmpty { context.insert(Team(name: name)) }
                    newTeamName = ""
                }
                Button("Cancel", role: .cancel) { newTeamName = "" }
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
