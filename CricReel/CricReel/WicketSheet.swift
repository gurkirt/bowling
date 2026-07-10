//
//  WicketSheet.swift
//  CricReel
//
//  Collects dismissal details for a preset wicket type and deterministically resolves
//  the batters at the crease afterwards. No dismissal dropdown — the type is chosen on
//  the pad. Strike is only asked for run-out / other; the over change is applied after
//  strike is assigned.
//

import SwiftUI

struct WicketSheet: View {
    let presetDismissal: DismissalType?
    /// Delivery type the wicket fell on (.none for a fair ball; wides/no-balls keep
    /// their penalty runs and don't advance the over).
    var extra: ExtraType = .none
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

    @State private var dismissedIsStriker = true
    @State private var fielderID: UUID?
    @State private var runsCompleted = 0
    @State private var newBatterID: UUID?
    @State private var strikerIsFirstResult = true

    private var dismissal: DismissalType { presetDismissal ?? .bowled }
    private var allOut: Bool { availableBatters.isEmpty }
    /// Wides and no-balls are not legal deliveries, so the over continues after them.
    private var advancesOver: Bool { extra != .wide && extra != .noBall }
    private var endsOver: Bool { isLastBallOfOver && advancesOver }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    HStack {
                        Image(systemName: "figure.fall").foregroundStyle(.red)
                        Text(dismissal.displayName).font(.headline)
                        if extra != .none {
                            Text("on a \(extra.displayName)")
                                .font(.subheadline).foregroundStyle(.orange)
                        }
                    }
                } footer: {
                    if !advancesOver {
                        Text("The \(extra.displayName.lowercased()) penalty runs still count and the ball is re-bowled.")
                    } else if extra != .none {
                        Text("Scored as \(extra.displayName.lowercased())s; the ball counts toward the over.")
                    }
                }

                if clipFilename != nil, let onDiscardClip {
                    Section {
                        Button(role: .destructive) { onDiscardClip(); dismiss() } label: {
                            Label("Not a delivery (discard clip)", systemImage: "trash")
                        }
                    }
                }

                if dismissal.asksStrike {
                    Section("Batter out") {
                        Picker("Batter out", selection: $dismissedIsStriker) {
                            Text(strikerName).tag(true)
                            Text(nonStrikerName).tag(false)
                        }
                        .pickerStyle(.segmented)
                    }
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
                    if allOut {
                        Text("All out — no batter remaining.").foregroundStyle(.secondary)
                    } else {
                        Picker("Coming in", selection: $newBatterID) {
                            ForEach(availableBatters, id: \.self) { id in
                                Text(lookup.name(id)).tag(UUID?.some(id))
                            }
                        }
                    }
                }

                if !allOut && dismissal.asksStrike {
                    Section(header: Text("On strike for next ball"),
                            footer: Text(endsOver ? "Over ends on this ball — strike rotates after this choice." : "")) {
                        let r = resultBatters()
                        Picker("Striker", selection: $strikerIsFirstResult) {
                            Text(lookup.name(r.0)).tag(true)
                            Text(lookup.name(r.1)).tag(false)
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
            .onAppear { newBatterID = availableBatters.first }
        }
    }

    /// Batters at the crease after runs + replacement, BEFORE any end-of-over swap.
    private func resultBatters() -> (UUID, UUID) {
        let crossed = runsCompleted % 2 == 1
        let atStrikerEnd = crossed ? nonStrikerID : strikerID
        let atNonStrikerEnd = crossed ? strikerID : nonStrikerID

        let outID = (dismissal.asksStrike && !dismissedIsStriker) ? nonStrikerID : strikerID
        let incoming = newBatterID ?? outID

        var sEnd = atStrikerEnd
        var nEnd = atNonStrikerEnd
        if outID == sEnd { sEnd = incoming } else if outID == nEnd { nEnd = incoming }
        return (sEnd, nEnd)
    }

    private func commit() {
        let (a, b) = resultBatters()
        var striker = dismissal.asksStrike ? (strikerIsFirstResult ? a : b) : a
        var nonStriker = dismissal.asksStrike ? (strikerIsFirstResult ? b : a) : b
        // Over change takes effect after strike is assigned — but only if this
        // delivery is legal (a wicket on a wide/no-ball doesn't end the over).
        if endsOver { swap(&striker, &nonStriker) }

        var input = BallInput()
        input.extra = extra
        input.padRuns = dismissal == .runOut ? runsCompleted : 0
        input.isWicket = true
        input.dismissal = dismissal
        input.dismissedPlayerID = (dismissal.asksStrike && !dismissedIsStriker) ? nonStrikerID : strikerID
        input.fielderID = dismissal.involvesFielder ? fielderID : nil
        input.newBatterID = allOut ? nil : newBatterID
        input.strikerAfterID = allOut ? nil : striker
        input.nonStrikerAfterID = allOut ? nil : nonStriker
        onCommit(input)
        dismiss()
    }
}
