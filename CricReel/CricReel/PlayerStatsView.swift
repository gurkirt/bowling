//
//  PlayerStatsView.swift
//  CricReel
//
//  Career journey for each player, aggregated across every match stored locally.
//

import SwiftUI
import SwiftData

struct PlayerStatsView: View {
    @Query(sort: \Player.name) private var players: [Player]
    @Query private var allDeliveries: [Delivery]

    private var lookup: PlayerLookup { PlayerLookup(players) }

    var body: some View {
        NavigationStack {
            List {
                if players.isEmpty {
                    ContentUnavailableView("No Players",
                                           systemImage: "chart.bar",
                                           description: Text("Career stats appear once matches are scored."))
                }
                let batting = StatsBuilder.batting(from: allDeliveries)
                let bowling = StatsBuilder.bowling(from: allDeliveries)
                ForEach(players) { player in
                    NavigationLink {
                        PlayerStatDetail(
                            name: player.name,
                            batting: batting[player.id],
                            bowling: bowling[player.id])
                    } label: {
                        HStack {
                            Text(player.name)
                            Spacer()
                            Text("\(batting[player.id]?.runs ?? 0) runs • \(bowling[player.id]?.wickets ?? 0) wkts")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Player Journeys")
        }
    }
}

private struct PlayerStatDetail: View {
    let name: String
    let batting: BattingLine?
    let bowling: BowlingLine?

    var body: some View {
        List {
            Section("Batting") {
                if let b = batting {
                    statRow("Runs", "\(b.runs)")
                    statRow("Balls faced", "\(b.ballsFaced)")
                    statRow("Fours", "\(b.fours)")
                    statRow("Sixes", "\(b.sixes)")
                    statRow("Strike rate", String(format: "%.1f", b.strikeRate))
                } else {
                    Text("No batting data yet.").foregroundStyle(.secondary)
                }
            }
            Section("Bowling") {
                if let b = bowling {
                    statRow("Overs", b.oversDisplay)
                    statRow("Runs conceded", "\(b.runsConceded)")
                    statRow("Wickets", "\(b.wickets)")
                    statRow("Economy", String(format: "%.1f", b.economy))
                } else {
                    Text("No bowling data yet.").foregroundStyle(.secondary)
                }
            }
        }
        .navigationTitle(name)
        .navigationBarTitleDisplayMode(.inline)
    }

    private func statRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).monospacedDigit().foregroundStyle(.secondary)
        }
    }
}
