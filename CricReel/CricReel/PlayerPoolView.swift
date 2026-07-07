//
//  PlayerPoolView.swift
//  CricReel
//
//  Manage the local pool of players (no accounts in the local-only phase).
//

import SwiftUI
import SwiftData

struct PlayerPoolView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \Player.name) private var players: [Player]

    @State private var showingAdd = false
    @State private var newName = ""
    @State private var newBatting = ""
    @State private var newBowling = ""

    var body: some View {
        NavigationStack {
            List {
                if players.isEmpty {
                    ContentUnavailableView("No Players Yet",
                                           systemImage: "person.3",
                                           description: Text("Add players to build teams and matches."))
                }
                ForEach(players) { player in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(player.name).font(.headline)
                        let sub = [player.battingStyle, player.bowlingStyle]
                            .compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " • ")
                        if !sub.isEmpty {
                            Text(sub).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete(perform: delete)
            }
            .navigationTitle("Players")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showingAdd = true } label: { Image(systemName: "plus") }
                }
            }
            .sheet(isPresented: $showingAdd) {
                addSheet
            }
        }
    }

    private var addSheet: some View {
        NavigationStack {
            Form {
                Section("Player") {
                    TextField("Name", text: $newName)
                    TextField("Batting style (optional)", text: $newBatting)
                    TextField("Bowling style (optional)", text: $newBowling)
                }
            }
            .navigationTitle("Add Player")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { resetAndClose() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save() }
                        .disabled(newName.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
    }

    private func save() {
        let player = Player(
            name: newName.trimmingCharacters(in: .whitespaces),
            battingStyle: newBatting.isEmpty ? nil : newBatting,
            bowlingStyle: newBowling.isEmpty ? nil : newBowling)
        context.insert(player)
        resetAndClose()
    }

    private func resetAndClose() {
        newName = ""; newBatting = ""; newBowling = ""
        showingAdd = false
    }

    private func delete(_ offsets: IndexSet) {
        for index in offsets { context.delete(players[index]) }
    }
}
