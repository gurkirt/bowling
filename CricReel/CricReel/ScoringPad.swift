//
//  ScoringPad.swift
//  CricReel
//
//  CricHeroes-style scoring keypad: big run buttons, extra modifiers (WD/NB/B/LB),
//  quick wicket buttons, and undo / swap-strike utilities.
//

import SwiftUI

struct ScoringPad: View {
    var enabled: Bool
    /// (extra, padRuns) — extra is `.none` for a normal delivery.
    var onScore: (ExtraType, Int) -> Void
    /// Quick dismissal, or nil to open the full wicket sheet ("More").
    var onWicket: (DismissalType?) -> Void
    var onUndo: () -> Void
    var onSwap: () -> Void

    @State private var armedExtra: ExtraType?

    private let runs = [0, 1, 2, 3, 4, 5, 6]
    private let columns = Array(repeating: GridItem(.flexible(), spacing: 8), count: 4)

    var body: some View {
        VStack(spacing: 10) {
            extrasRow
            if let armed = armedExtra {
                Text("\(armed.displayName) selected — tap the runs taken (0 if none)")
                    .font(.caption).foregroundStyle(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            runsGrid
            wicketRow
            utilityRow
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
        .opacity(enabled ? 1 : 0.4)
        .allowsHitTesting(enabled)
        .animation(.easeInOut(duration: 0.15), value: armedExtra)
    }

    // MARK: - Extras

    private var extrasRow: some View {
        HStack(spacing: 8) {
            ForEach([ExtraType.wide, .noBall, .bye, .legBye]) { extra in
                Button {
                    armedExtra = (armedExtra == extra) ? nil : extra
                } label: {
                    Text(extra.shortLabel)
                        .font(.subheadline.bold())
                        .frame(maxWidth: .infinity, minHeight: 40)
                }
                .buttonStyle(PadButtonStyle(
                    fill: armedExtra == extra ? Color.orange : Color.orange.opacity(0.18),
                    fg: armedExtra == extra ? .white : .orange))
            }
        }
    }

    // MARK: - Runs

    private var runsGrid: some View {
        LazyVGrid(columns: columns, spacing: 8) {
            ForEach(runs, id: \.self) { r in
                Button {
                    onScore(armedExtra ?? .none, r)
                    armedExtra = nil
                } label: {
                    Text("\(r)")
                        .font(.title.bold())
                        .frame(maxWidth: .infinity, minHeight: 60)
                }
                .buttonStyle(PadButtonStyle(fill: runFill(r), fg: runFg(r)))
            }
        }
    }

    private func runFill(_ r: Int) -> Color {
        switch r {
        case 4: return .blue
        case 6: return .purple
        default: return Color(.secondarySystemBackground)
        }
    }
    private func runFg(_ r: Int) -> Color { (r == 4 || r == 6) ? .white : .primary }

    // MARK: - Wicket

    private var wicketRow: some View {
        HStack(spacing: 8) {
            wicketButton("Bowled", .bowled)
            wicketButton("Caught", .caught)
            wicketButton("Run Out", .runOut)
            Button { onWicket(nil) } label: {
                Text("OUT…").font(.subheadline.bold())
                    .frame(maxWidth: .infinity, minHeight: 46)
            }
            .buttonStyle(PadButtonStyle(fill: .red, fg: .white))
        }
    }

    private func wicketButton(_ title: String, _ type: DismissalType) -> some View {
        Button { onWicket(type) } label: {
            Text(title).font(.subheadline.bold())
                .frame(maxWidth: .infinity, minHeight: 46)
        }
        .buttonStyle(PadButtonStyle(fill: Color.red.opacity(0.15), fg: .red))
    }

    // MARK: - Utility

    private var utilityRow: some View {
        HStack(spacing: 8) {
            Button { onUndo() } label: {
                Label("Undo", systemImage: "arrow.uturn.backward")
                    .frame(maxWidth: .infinity, minHeight: 44)
            }
            .buttonStyle(PadButtonStyle(fill: Color(.secondarySystemBackground), fg: .primary))
            Button { onSwap() } label: {
                Label("Swap Strike", systemImage: "arrow.left.arrow.right")
                    .frame(maxWidth: .infinity, minHeight: 44)
            }
            .buttonStyle(PadButtonStyle(fill: Color(.secondarySystemBackground), fg: .primary))
        }
    }
}

private struct PadButtonStyle: ButtonStyle {
    var fill: Color
    var fg: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(fg)
            .background(fill, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.black.opacity(0.05)))
            .scaleEffect(configuration.isPressed ? 0.95 : 1)
            .animation(.easeOut(duration: 0.1), value: configuration.isPressed)
    }
}
