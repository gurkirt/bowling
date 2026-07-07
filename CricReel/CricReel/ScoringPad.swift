//
//  ScoringPad.swift
//  CricReel
//
//  The scoring keypad used on the ball-entry screen: run buttons, extra modifiers
//  (WD/NB/B/LB armed then tap runs) and all dismissal types.
//

import SwiftUI

struct ScoringPad: View {
    /// (extra, padRuns) — extra is `.none` for a normal delivery.
    var onScore: (ExtraType, Int) -> Void
    var onWicket: (DismissalType) -> Void

    @State private var armedExtra: ExtraType?

    private let runs = [0, 1, 2, 3, 4, 5, 6]
    private let runColumns = Array(repeating: GridItem(.flexible(), spacing: 8), count: 4)
    private let outColumns = Array(repeating: GridItem(.flexible(), spacing: 8), count: 3)

    var body: some View {
        VStack(spacing: 16) {
            VStack(spacing: 10) {
                extrasRow
                if let armed = armedExtra {
                    Text("\(armed.displayName) selected — tap the runs taken (0 if none)")
                        .font(.caption).foregroundStyle(.orange)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                runsGrid
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("WICKET").font(.caption).foregroundStyle(.secondary)
                LazyVGrid(columns: outColumns, spacing: 8) {
                    ForEach(DismissalType.allCases) { type in
                        Button { onWicket(type) } label: {
                            Text(type.displayName).font(.subheadline.bold())
                                .frame(maxWidth: .infinity, minHeight: 46)
                        }
                        .buttonStyle(PadButtonStyle(fill: Color.red.opacity(0.14), fg: .red))
                    }
                }
            }
        }
    }

    private var extrasRow: some View {
        HStack(spacing: 8) {
            ForEach([ExtraType.wide, .noBall, .bye, .legBye]) { extra in
                Button {
                    armedExtra = (armedExtra == extra) ? nil : extra
                } label: {
                    Text(extra.shortLabel).font(.subheadline.bold())
                        .frame(maxWidth: .infinity, minHeight: 42)
                }
                .buttonStyle(PadButtonStyle(
                    fill: armedExtra == extra ? Color.orange : Color.orange.opacity(0.16),
                    fg: armedExtra == extra ? .white : .orange))
            }
        }
    }

    private var runsGrid: some View {
        LazyVGrid(columns: runColumns, spacing: 8) {
            ForEach(runs, id: \.self) { r in
                Button {
                    onScore(armedExtra ?? .none, r)
                    armedExtra = nil
                } label: {
                    Text("\(r)").font(.title.bold())
                        .frame(maxWidth: .infinity, minHeight: 64)
                }
                .buttonStyle(PadButtonStyle(fill: Color(.secondarySystemBackground), fg: .primary))
            }
        }
    }
}

struct PadButtonStyle: ButtonStyle {
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
