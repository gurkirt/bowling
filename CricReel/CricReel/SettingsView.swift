//
//  SettingsView.swift
//  CricReel
//
//  Recording + detection settings.
//

import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings = SettingsStore.shared

    var body: some View {
        NavigationStack {
            Form {
                Section("Appearance") {
                    Picker("Theme", selection: $settings.appearance) {
                        ForEach(AppAppearance.allCases) { Text($0.label).tag($0) }
                    }
                    .pickerStyle(.segmented)
                }

                Section(header: Text("Recording"),
                        footer: Text("Detection always runs at 30 fps to save battery — higher frame rates only affect the recorded clips.")) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Resolution").font(.subheadline).foregroundStyle(.secondary)
                        Picker("Resolution", selection: $settings.resolution) {
                            ForEach(VideoResolution.allCases) { Text($0.displayName).tag($0) }
                        }
                        .pickerStyle(.segmented)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Frame rate").font(.subheadline).foregroundStyle(.secondary)
                        Picker("Frame rate", selection: $settings.frameRate) {
                            ForEach(FrameRateOption.allCases) { Text($0.displayName).tag($0) }
                        }
                        .pickerStyle(.segmented)
                    }
                }

                Section("Clip length") {
                    stepperRow("Pre-trigger", value: $settings.preTriggerDuration,
                               range: 0.5...3, step: 0.5, unit: "s")
                    stepperRow("Post-trigger", value: $settings.postTriggerDuration,
                               range: 1...5, step: 0.5, unit: "s")
                    stepperRow("Cooldown", value: $settings.cooldownDuration,
                               range: 2...20, step: 1, unit: "s")
                }

                Section(header: Text("Detection"),
                        footer: Text("Higher threshold and required-frame count reduce false triggers but may miss deliveries.")) {
                    HStack {
                        Text("Score threshold")
                        Spacer()
                        Text(settings.scoreThreshold, format: .number.precision(.fractionLength(2)))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $settings.scoreThreshold, in: 0.3...0.9, step: 0.05)
                    Stepper("Window: \(settings.windowSize) frames",
                            value: $settings.windowSize, in: 4...16)
                    Stepper("Required: \(settings.requiredActionCount) frames",
                            value: $settings.requiredActionCount, in: 2...settings.windowSize)
                }
            }
            .navigationTitle("Settings")
        }
    }

    private func stepperRow(_ label: String, value: Binding<Double>,
                            range: ClosedRange<Double>, step: Double, unit: String) -> some View {
        Stepper(value: value, in: range, step: step) {
            HStack {
                Text(label)
                Spacer()
                Text("\(value.wrappedValue, format: .number.precision(.fractionLength(1)))\(unit)")
                    .foregroundStyle(.secondary)
            }
        }
    }
}
