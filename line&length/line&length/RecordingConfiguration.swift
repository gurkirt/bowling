//
//  RecordingConfiguration.swift
//  line&length
//
//  Created by GitHub Copilot on 02.11.2025.
//

import Foundation
import AVFoundation

enum VideoResolution: String, CaseIterable, Identifiable {
    case hd1080 = "1080p HD"
    
    var id: String { rawValue }
    
    var displayName: String { rawValue }
    
    var dimensions: (width: Int, height: Int) {
        switch self {
        case .hd1080:
            return (1920, 1080)
        }
    }
    
    var sessionPreset: AVCaptureSession.Preset {
        switch self {
        case .hd1080:
            return .hd1920x1080
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
        case .fps30:
            return 30.0
        case .fps60:
            return 60.0
        }
    }
    
    var frameDuration: CMTime {
        CMTime(seconds: 1.0 / value, preferredTimescale: 600)
    }
}

struct RecordingConfiguration: Equatable {
    let resolution: VideoResolution
    let frameRate: FrameRateOption
    let preTriggerDuration: TimeInterval
    let postTriggerDuration: TimeInterval
    
    init(resolution: VideoResolution,
         frameRate: FrameRateOption,
         preTriggerDuration: TimeInterval = 1.0,
         postTriggerDuration: TimeInterval = 2.0) {
        self.resolution = resolution
        self.frameRate = frameRate
        self.preTriggerDuration = preTriggerDuration
        self.postTriggerDuration = postTriggerDuration
    }
    
    static let `default` = RecordingConfiguration(resolution: .hd1080, frameRate: .fps60)
    
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
