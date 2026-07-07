//
//  BallConfirmSheet.swift
//  CricReel
//
//  Ball outcome entry. Presented either after an auto-detected clip (with a preview +
//  a "false trigger" discard option) or manually (no clip). Also the bowler picker.
//

import SwiftUI
import AVKit

struct BallEntrySheet: View {
    let clipFilename: String?
    let strikerName: String
    let strikerID: UUID?
    let nonStrikerName: String
    let nonStrikerID: UUID?
    var onCommit: (_ input: BallInput, _ keepClip: Bool) -> Void
    var onDiscard: () -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var runs = 0
    @State private var isWide = false
    @State private var isWicket = false
    @State private var dismissal: DismissalType = .bowled
    @State private var dismissedIsStriker = true

    private let runOptions = [0, 1, 2, 3, 4, 5, 6]

    var body: some View {
        NavigationStack {
            Form {
                if let clipFilename {
                    Section("Detected Clip") {
                        VideoPlayer(player: AVPlayer(url: ClipStore.url(forClip: clipFilename)))
                            .frame(height: 200)
                            .listRowInsets(EdgeInsets())
                        Button(role: .destructive) {
                            onDiscard()
                        } label: {
                            Label("Not a delivery (discard clip)", systemImage: "trash")
                        }
                    }
                }

                Section("Runs off the bat") {
                    Picker("Runs", selection: $runs) {
                        ForEach(runOptions, id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(isWide)
                }

                Section("Extras") {
                    Toggle("Wide", isOn: $isWide)
                }

                Section("Wicket") {
                    Toggle("Wicket falls", isOn: $isWicket)
                    if isWicket {
                        Picker("How out", selection: $dismissal) {
                            ForEach(DismissalType.allCases, id: \.self) {
                                Text($0.displayName).tag($0)
                            }
                        }
                        Picker("Batter out", selection: $dismissedIsStriker) {
                            Text("\(strikerName) (striker)").tag(true)
                            Text("\(nonStrikerName) (non-striker)").tag(false)
                        }
                    }
                }
            }
            .navigationTitle("Ball Outcome")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        // Cancelling an auto entry keeps the clip on disk; a manual entry just closes.
                        dismiss()
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { commit() }
                }
            }
        }
    }

    private func commit() {
        var input = BallInput()
        if isWide {
            input.extraType = .wide
            input.runsOffBat = 0
        } else {
            input.runsOffBat = runs
        }
        input.isWicket = isWicket
        if isWicket {
            input.dismissalType = dismissal
            input.dismissedPlayerID = dismissedIsStriker ? strikerID : nonStrikerID
        }
        onCommit(input, clipFilename != nil)
    }
}

struct BowlerPickerSheet: View {
    let bowlerIDs: [UUID]
    let lookup: PlayerLookup
    let current: UUID?
    var onSelect: (UUID) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List(bowlerIDs, id: \.self) { id in
                Button {
                    onSelect(id)
                    dismiss()
                } label: {
                    HStack {
                        Text(lookup.name(id)).foregroundStyle(.primary)
                        Spacer()
                        if id == current {
                            Image(systemName: "checkmark").foregroundStyle(.blue)
                        }
                    }
                }
            }
            .navigationTitle("Select Bowler")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
