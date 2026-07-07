//
//  VideoWriter.swift
//  CriClips
//
//  Maintains a pre-trigger ring buffer of pixel frames.  On a trigger it stitches
//  pre-trigger frames + post-trigger frames into a single H.264 .mp4 written to
//  the Documents directory.  After writing completes a cooldown timer must expire
//  and the pre-trigger buffer must refill before the next trigger is armed.
//

import AVFoundation
import CoreVideo

private struct BufferedFrame {
    let pixelBuffer: CVPixelBuffer
    let timestamp: CMTime
}

class VideoWriter: ObservableObject {
    @Published var isReadyForTrigger = false
    @Published var isRecording = false
    @Published var isCoolingDown = false
    @Published var cooldownRemaining: Int = 0  // whole seconds remaining
    @Published var lastSavedURL: URL?
    @Published var lastError: String?

    private var config = RecordingConfiguration.default

    private var preTriggerTarget: Int { config.preTriggerFrameCount }
    private var postTriggerTarget: Int { config.postTriggerFrameCount }

    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private let writingQueue = DispatchQueue(label: "criclips.video.write", qos: .userInitiated)

    private var pixelBufferPool: CVPixelBufferPool?
    private var videoWidth = 1920
    private var videoHeight = 1080
    private var pixelFormat: OSType = kCVPixelFormatType_32BGRA

    private var preTriggerBuffer: [BufferedFrame] = []
    private var postTriggerBuffer: [BufferedFrame] = []
    private var isWriting = false
    private var isCollectingPost = false
    private var postCollectedCount = 0

    // Cooldown: frames are buffered, but trigger is locked until cooldownEndTime has passed.
    private var cooldownEndTime: Date = .distantPast
    private var cooldownTimer: Timer?
    private var sessionStartTime: Date?

    // MARK: - Lifecycle

    func startCamera() {
        writingQueue.async { [weak self] in
            guard let self else { return }
            self.cleanup()
            self.sessionStartTime = Date()
            DispatchQueue.main.async { self.isReadyForTrigger = false }
        }
    }

    func updateConfiguration(_ newConfig: RecordingConfiguration) {
        writingQueue.async { [weak self] in
            guard let self else { return }
            self.config = newConfig
            self.cleanup()
            self.pixelBufferPool = nil
            let dims = newConfig.resolution.dimensions
            self.videoWidth  = dims.width
            self.videoHeight = dims.height
            self.sessionStartTime = Date()
            DispatchQueue.main.async { self.isReadyForTrigger = false }
        }
    }

    // MARK: - Trigger

    func triggerRecording() {
        writingQueue.async { [weak self] in
            guard let self else { return }
            guard !self.isWriting, !self.isCollectingPost else { return }
            guard self.isReadyForTrigger else { return }
            guard self.preTriggerBuffer.count >= self.preTriggerTarget else { return }

            DispatchQueue.main.async {
                self.isReadyForTrigger = false
                self.isRecording = true
            }
            self.isCollectingPost = true
            self.postTriggerBuffer.removeAll()
            self.postCollectedCount = 0
        }
    }

    // MARK: - Frame ingestion

    func addFrame(_ sampleBuffer: CMSampleBuffer) {
        writingQueue.async { [weak self] in
            guard let self else { return }
            guard !self.isWriting else { return }

            guard let src = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let ts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

            // Initialise pool on first frame
            if self.pixelBufferPool == nil {
                let w = CVPixelBufferGetWidth(src)
                let h = CVPixelBufferGetHeight(src)
                let fmt = CVPixelBufferGetPixelFormatType(src)
                self.videoWidth  = w
                self.videoHeight = h
                self.pixelFormat = fmt
                guard self.createPool(width: w, height: h, format: fmt) else { return }
            }

            guard let copy = self.copyPixelBuffer(src) else { return }
            let frame = BufferedFrame(pixelBuffer: copy, timestamp: ts)

            if self.isCollectingPost {
                self.postTriggerBuffer.append(frame)
                self.postCollectedCount += 1
                if self.postCollectedCount >= self.postTriggerTarget {
                    self.isCollectingPost = false
                    self.writeClip()
                }
            } else {
                self.preTriggerBuffer.append(frame)
                if self.preTriggerBuffer.count > self.preTriggerTarget {
                    self.preTriggerBuffer.removeFirst()
                }
                // Arm trigger when buffer is full AND cooldown has expired
                if !self.isReadyForTrigger,
                   self.preTriggerBuffer.count >= self.preTriggerTarget,
                   Date() >= self.cooldownEndTime,
                   let start = self.sessionStartTime,
                   Date().timeIntervalSince(start) >= 0.5 {
                    DispatchQueue.main.async { self.isReadyForTrigger = true }
                }
            }
        }
    }

    // MARK: - Video Writing

    private func writeClip() {
        isWriting = true
        let allFrames = preTriggerBuffer + postTriggerBuffer

        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let name = "criclip_\(fmt.string(from: Date())).mp4"
        let outURL = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(name)

        do {
            assetWriter = try AVAssetWriter(outputURL: outURL, fileType: .mp4)

            let videoSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: videoWidth,
                AVVideoHeightKey: videoHeight,
                AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: targetBitRate()]
            ]

            videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            videoInput?.expectsMediaDataInRealTime = false
            videoInput?.transform = .identity   // pixel data is already portrait (videoRotationAngle=90 rotates it)

            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: videoInput!,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                    kCVPixelBufferWidthKey  as String: videoWidth,
                    kCVPixelBufferHeightKey as String: videoHeight
                ])
            self.adaptor = adaptor

            guard assetWriter!.canAdd(videoInput!) else { failWrite("Cannot add video input"); return }
            assetWriter!.add(videoInput!)
            guard assetWriter!.startWriting() else {
                failWrite(assetWriter?.error?.localizedDescription ?? "startWriting failed")
                return
            }

            assetWriter!.startSession(atSourceTime: .zero)
            let tpf = config.frameRate.frameDuration
            var t = CMTime.zero
            var written = 0

            for frame in allFrames {
                while videoInput?.isReadyForMoreMediaData == false {
                    Thread.sleep(forTimeInterval: 0.001)
                }
                if adaptor.append(frame.pixelBuffer, withPresentationTime: t) {
                    written += 1
                }
                t = CMTimeAdd(t, tpf)
            }
            videoInput!.markAsFinished()

            assetWriter!.finishWriting { [weak self] in
                guard let self else { return }
                let err = self.assetWriter?.error
                self.isWriting = false
                if let err {
                    print("❌ [CriClips] VideoWriter: \(err)")
                    DispatchQueue.main.async { self.lastError = err.localizedDescription }
                } else {
                    print("✅ [CriClips] Saved \(name) (\(written) frames)")
                    DispatchQueue.main.async {
                        self.lastSavedURL = outURL
                        NotificationCenter.default.post(name: .newClipSaved, object: outURL)
                    }
                }
                self.beginCooldown()
            }

        } catch {
            failWrite(error.localizedDescription)
        }
    }

    // MARK: - Cooldown

    private func beginCooldown() {
        let duration = config.cooldownDuration
        cooldownEndTime = Date().addingTimeInterval(duration)

        // Clear pre-trigger buffer so it refills during cooldown
        writingQueue.async { [weak self] in
            self?.preTriggerBuffer.removeAll()
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.isRecording = false
            self.isCoolingDown = true
            self.cooldownRemaining = Int(ceil(duration))

            // Tick every second
            self.cooldownTimer?.invalidate()
            self.cooldownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
                guard let self else { timer.invalidate(); return }
                let remaining = max(0, self.cooldownEndTime.timeIntervalSinceNow)
                self.cooldownRemaining = Int(ceil(remaining))
                if remaining <= 0 {
                    timer.invalidate()
                    self.isCoolingDown = false
                    // isReadyForTrigger will be set by addFrame once the buffer refills
                }
            }
            RunLoop.main.add(self.cooldownTimer!, forMode: .common)
        }
    }

    // MARK: - Error Handling

    private func failWrite(_ msg: String) {
        print("❌ [CriClips] VideoWriter: \(msg)")
        isWriting = false
        isCollectingPost = false
        DispatchQueue.main.async { [weak self] in
            self?.isRecording = false
            self?.lastError = msg
            self?.isReadyForTrigger = true
        }
    }

    // MARK: - Pixel Buffer Pool

    private func createPool(width: Int, height: Int, format: OSType) -> Bool {
        let poolAttrs  = [kCVPixelBufferPoolMinimumBufferCountKey: 60] as CFDictionary
        let bufAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey  as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        let ret = CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs,
                                          bufAttrs as CFDictionary, &pixelBufferPool)
        return ret == kCVReturnSuccess
    }

    private func copyPixelBuffer(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        guard let pool = pixelBufferPool else { return nil }
        var dst: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &dst) == kCVReturnSuccess,
              let out = dst else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(out, [])
        defer {
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
            CVPixelBufferUnlockBaseAddress(out, [])
        }
        if let s = CVPixelBufferGetBaseAddress(src), let d = CVPixelBufferGetBaseAddress(out) {
            let bytes = CVPixelBufferGetDataSize(src)
            memcpy(d, s, bytes)
        }
        return out
    }

    // MARK: - Helpers

    private func targetBitRate() -> Int {
        let base = 16_000_000.0
        return max(8_000_000, Int(base * config.frameRate.value / 30.0))
    }

    private func cleanup() {
        preTriggerBuffer.removeAll()
        postTriggerBuffer.removeAll()
        isWriting = false
        isCollectingPost = false
        postCollectedCount = 0
    }
}
