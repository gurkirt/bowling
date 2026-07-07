//
//  SettingsStore.swift
//  CricReel
//
//  Persisted recording + detection settings (UserDefaults), shared across the app.
//

import Foundation
import Combine

final class SettingsStore: ObservableObject {
    static let shared = SettingsStore()

    @Published var resolution: VideoResolution { didSet { persist() } }
    @Published var frameRate: FrameRateOption { didSet { persist() } }
    @Published var preTriggerDuration: Double { didSet { persist() } }
    @Published var postTriggerDuration: Double { didSet { persist() } }
    @Published var cooldownDuration: Double { didSet { persist() } }

    // Detection tuning
    @Published var scoreThreshold: Double { didSet { persist() } }
    @Published var windowSize: Int { didSet { persist() } }
    @Published var requiredActionCount: Int { didSet { persist() } }

    private let d = UserDefaults.standard

    private init() {
        resolution = VideoResolution(rawValue: d.string(forKey: "res") ?? "") ?? .hd1080
        frameRate = FrameRateOption(rawValue: d.string(forKey: "fps") ?? "") ?? .fps30
        preTriggerDuration = d.object(forKey: "pre") as? Double ?? 1.0
        postTriggerDuration = d.object(forKey: "post") as? Double ?? 2.0
        cooldownDuration = d.object(forKey: "cool") as? Double ?? 8.0
        scoreThreshold = d.object(forKey: "thr") as? Double ?? 0.5
        windowSize = d.object(forKey: "win") as? Int ?? 8
        requiredActionCount = d.object(forKey: "req") as? Int ?? 4
    }

    private func persist() {
        d.set(resolution.rawValue, forKey: "res")
        d.set(frameRate.rawValue, forKey: "fps")
        d.set(preTriggerDuration, forKey: "pre")
        d.set(postTriggerDuration, forKey: "post")
        d.set(cooldownDuration, forKey: "cool")
        d.set(scoreThreshold, forKey: "thr")
        d.set(windowSize, forKey: "win")
        d.set(requiredActionCount, forKey: "req")
    }

    var recordingConfiguration: RecordingConfiguration {
        RecordingConfiguration(
            resolution: resolution,
            frameRate: frameRate,
            preTriggerDuration: preTriggerDuration,
            postTriggerDuration: postTriggerDuration,
            cooldownDuration: cooldownDuration)
    }
}
