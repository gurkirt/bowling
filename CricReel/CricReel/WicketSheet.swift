//
//  WicketSheet.swift
//  CricReel
//
//  Collects dismissal details and deterministically resolves the batters at the crease
//  afterwards (handles run-outs, runs completed, new batter, and end-of-over strike).
//

import SwiftUI

struct WicketSheet: View {
    let presetDismissal: DismissalType?
    let strikerID: UUID
    let nonStrikerID: UUID
    let strikerName: String
    let nonStrikerName: String
    let fieldingSideIDs: [UUID]
    let availableBatters: [UUID]
    let isLastBallOfOver: Bool
    let lookup: PlayerLookup
    let clipFilename: String?

    var onCommit: (BallInput) -> Void
    var onCancel: () -> Void
    var onDiscardClip: (() -> Void)?

    @Environment(\.dismiss) private var dismiss

    @State private var dismissal: DismissalType = .bowled
    @State private var dismissedIsStriker = true
    @State private var fielderID: UUID?
    @State private var runsCompleted = 0
    @State private var newBatterID: UUID?
    @State private var strikerIsFirstResult = true

    var body: some View {
        NavigationStack {
            Form {
                if clipFilename != nil {
                    Section("Detected Clip") {
                        Label("Clip will be attached to this ball", systemImage: "film")
                            .font(.caption)
                        if let onDiscardClip {
                            Button(role: .destructive) { onDiscardClip(); dismiss() } label: {
                                Label("Not a delivery (discard clip)", systemImage: "trash")
                            }
                        }
                    }
                }

                Section("Dismissal") {
                    Picker("How out", selection: $dismissal) {
                        ForEach(DismissalType.allCases) { Text($0.displayName).tag($0) }
                    }
                    Picker("Batter out", selection: $dismissedIsStriker) {
                        Text(strikerName).tag(true)
                        Text(nonStrikerName).tag(false)
                    }
                    .pickerStyle(.segmented)
                    .disabled(dismissal != .runOut)  // only run-out can dismiss the non-striker
                }

                if dismissal.involvesFielder {
                    Section(dismissal == .runOut ? "Run out by" : "Fielder") {
                        Picker("Fielder", selection: $fielderID) {
                            Text("—").tag(UUID?.none)
                            ForEach(fieldingSideIDs, id: \.self) { id in
                                Text(lookup.name(id)).tag(UUID?.some(id))
                            }
                        }
                    }
                }

                if dismissal == .runOut {
                    Section("Runs completed before run-out") {
                        Stepper("\(runsCompleted) run\(runsCompleted == 1 ? "" : "s")",
                                value: $runsCompleted, in: 0...6)
                    }
                }

                Section("New batter") {
                    if availableBatters.isEmpty {
                        Text("All out — no batter remaining.").foregroundStyle(.secondary)
                    } else {
                        Picker("Coming in", selection: $newBatterID) {
                            ForEach(availableBatters, id: \.self) { id in
                                Text(lookup.name(id)).tag(UUID?.some(id))
                            }
                        }
                    }
                }

                if !availableBatters.isEmpty {
                    Section(header: Text("On strike for next ball"),
                            footer: Text(isLastBallOfOver ? "Over ends on this ball — strike has been rotated. Adjust if needed." : "Adjust if the batters crossed.")) {
                        Picker("Striker", selection: $strikerIsFirstResult) {
                            Text(lookup.name(resultBatters().0)).tag(true)
                            Text(lookup.name(resultBatters().1)).tag(false)
                        }
                        .pickerStyle(.segmented)
                    }
                }
            }
            .navigationTitle("Wicket")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { onCancel(); dismiss() } }
                ToolbarItem(placement: .confirmationAction) { Button("Save") { commit() } }
            }
            .onAppear {
                if let preset = presetDismissal { dismissal = preset }
                if dismissal != .runOut { dismissedIsStriker = true }
                newBatterID = availableBatters.first
            }
            .onChange(of: dismissal) { _, new in
                if new != .runOut { dismissedIsStriker = true; runsCompleted = 0 }
            }
        }
    }

    /// The two batters at the crease after the dismissal, before choosing who's on strike.
    /// Returns (defaultStrikerEnd, defaultNonStrikerEnd).
    private func resultBatters() -> (UUID, UUID) {
        // Positions after the runs completed.
        let crossed = runsCompleted % 2 == 1
        let atStrikerEnd = crossed ? nonStrikerID : strikerID
        let atNonStrikerEnd = crossed ? strikerID : nonStrikerID

        let outID = dismissedIsStriker ? strikerID : nonStrikerID
        let incoming = newBatterID ?? outID

        var sEnd = atStrikerEnd
        var nEnd = atNonStrikerEnd
        if outID == sEnd { sEnd = incoming } else if outID == nEnd { nEnd = incoming }

        if isLastBallOfOver { swap(&sEnd, &nEnd) }
        return (sEnd, nEnd)
    }

    private func commit() {
        let (a, b) = resultBatters()
        let striker = strikerIsFirstResult ? a : b
        let nonStriker = strikerIsFirstResult ? b : a

        var input = BallInput()
        input.extra = .none
        input.padRuns = dismissal == .runOut ? runsCompleted : 0
        input.isWicket = true
        input.dismissal = dismissal
        input.dismissedPlayerID = dismissedIsStriker ? strikerID : nonStrikerID
        input.fielderID = dismissal.involvesFielder ? fielderID : nil
        input.newBatterID = availableBatters.isEmpty ? nil : newBatterID
        input.strikerAfterID = availableBatters.isEmpty ? nil : striker
        input.nonStrikerAfterID = availableBatters.isEmpty ? nil : nonStriker
        onCommit(input)
        dismiss()
    }
}
