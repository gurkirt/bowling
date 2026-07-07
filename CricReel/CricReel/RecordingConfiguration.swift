//
//  RecordingConfiguration.swift
//  CriClips
//

import Foundation
import AVFoundation

enum VideoResolution: String, CaseIterable, Identifiable {
    case hd1080 = "1080p HD"
    case uhd4k  = "4K UHD"

    var id: String { rawValue }
    var displayName: String { rawValue }

    /// Landscape pixel dimensions as reported by AVCaptureDevice.formats.
    var dimensions: (width: Int, height: Int) {
        switch self {
        case .hd1080: return (1920, 1080)
        case .uhd4k:  return (3840, 2160)
        }
    }

    var sessionPreset: AVCaptureSession.Preset {
        switch self {
        case .hd1080: return .hd1920x1080
        case .uhd4k:  return .hd4K3840x2160
        }
    }
}

enum FrameRateOption: String, CaseIterable, Identifiable {
    case fps30 = "30 fps"
    case fps60 = "60 fps"

    var id: String { rawValue }
    var displayName: String { rawValue }

    var value: Double {
        switch self {
        case .fps30: return 30.0
        case .fps60: return 60.0
        }
    }

    var frameDuration: CMTime {
        CMTime(seconds: 1.0 / value, preferredTimescale: 600)
    }
}

struct RecordingConfiguration: Equatable {
    let resolution: VideoResolution
    let frameRate: FrameRateOption
    /// Seconds of footage kept before the trigger fires.
    let preTriggerDuration: TimeInterval
    /// Seconds of footage captured after the trigger fires.
    let postTriggerDuration: TimeInterval
    /// Minimum wait (seconds) after a clip is saved before the next trigger is armed.
    let cooldownDuration: TimeInterval

    init(
        resolution: VideoResolution = .hd1080,
        frameRate: FrameRateOption = .fps30,
        preTriggerDuration: TimeInterval = 1.0,
        postTriggerDuration: TimeInterval = 2.0,
        cooldownDuration: TimeInterval = 10.0
    ) {
        self.resolution = resolution
        self.frameRate = frameRate
        self.preTriggerDuration = preTriggerDuration
        self.postTriggerDuration = postTriggerDuration
        self.cooldownDuration = cooldownDuration
    }

    static let `default` = RecordingConfiguration()

    var preTriggerFrameCount: Int {
        max(1, Int(round(frameRate.value * preTriggerDuration)))
    }

    var postTriggerFrameCount: Int {
        max(1, Int(round(frameRate.value * postTriggerDuration)))
    }

    var totalBufferSize: Int {
        preTriggerFrameCount + postTriggerFrameCount + 2
    }
}
