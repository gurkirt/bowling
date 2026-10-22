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
                    ContentUnavailableView {
                        Label("No Teams Yet", systemImage: "person.3.sequence")
                    } description: {
                        Text("Create teams manually, or load the sample league to try things out.")
                    } actions: {
                        Button { seedSampleTeams() } label: {
                            Label("Add Sample Teams", systemImage: "square.stack.3d.up.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        Button { showingNew = true } label: { Text("New Team") }
                    }
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
                    Menu {
                        Button { showingNew = true } label: { Label("New Team", systemImage: "plus") }
                        Button { seedSampleTeams() } label: {
                            Label("Add Sample Teams", systemImage: "square.stack.3d.up.fill")
                        }
                    } label: { Image(systemName: "plus") }
                }
            }
            .sheet(isPresented: $showingNew) { NewTeamSheet() }
        }
    }

    // MARK: - Sample data

    /// (team name, reel name, ordered squad of full player names).
    private static let sampleTeams: [(name: String, reel: String, players: [String])] = [
        ("Kerala Warriors", "KW", [
            "Joys Rajan", "Deepu Stephen", "Jackson Kallickal", "Rajesh Pattupara",
            "Shanod Jacob", "Shijo Joseph", "Ajay Babu", "Jeethu Kalarickal",
            "Lynn Joe Abraham", "Ajay Babu"]),
        ("Zurich Mavericks Adliswil", "ZMA", [
            "Anil Kumar", "Brighton Thomas", "Suhash Venkatesh", "Aliullah",
            "Maqbool S", "Irfan Basha", "Vijay Gulati", "Karan K",
            "Vinit Pagaria", "Gurkirt Singh"]),
        ("Zurich Master Blasters", "ZMB", [
            "Rohit Ranjan", "Sreekanth", "Vineesh Vasudevan", "Amit Koul",
            "Mukesh Kukkerabettu", "Nitin Paricharak", "Vijesh", "Vishal Prakash",
            "Kasigopi", "Shivank Sharma"]),
        ("Glatt Gladiators", "GG", [
            "Ashish Pasrija", "Akhil Paul", "Deepak Moghadas Sathyaprakash", "Samsoor Safi",
            "Sunil S", "Abdulqahar Popalzai", "Malyar Stanikzai", "Feeda Khan",
            "Sandeep Chourasia", "Rahul Singh"])
    ]

    /// Reel name for a player = first name, truncated to 8 characters.
    private func reelName(for fullName: String) -> String {
        let first = fullName.split(separator: " ").first.map(String.init) ?? fullName
        return String(first.prefix(8))
    }

    /// Insert the sample league. Skips any team whose name already exists so the
    /// button is safe to tap more than once.
    private func seedSampleTeams() {
        for t in Self.sampleTeams where !teams.contains(where: { $0.name == t.name }) {
            let team = Team(name: t.name, reelName: t.reel)
            context.insert(team)
            for rawName in t.players {
                let name = rawName.trimmingCharacters(in: .whitespaces)
                let player = Player(name: name, reelName: reelName(for: name))
                context.insert(player)
                team.players.append(player)
            }
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
