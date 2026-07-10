//
//  VideoWriter.swift
//  CriClips
//
//  Maintains a pre-trigger ring buffer of pixel frames.  On a trigger it opens an
//  AVAssetWriter immediately, drains the pre-trigger frames into the hardware H.264
//  encoder, then streams every post-trigger frame straight from the camera into the
//  encoder — nothing is buffered in memory, so post-trigger duration is unbounded.
//  After writing completes a cooldown timer must expire and the pre-trigger buffer
//  must refill before the next trigger is armed.
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

    /// Writer lifecycle. All transitions happen on writingQueue.
    private enum State {
        case buffering      // filling the pre-trigger ring, waiting for a trigger
        case streamingPost  // writer open, camera frames go straight to the encoder
        case finishing      // markAsFinished called, waiting for finishWriting
    }
    private var state: State = .buffering

    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var currentClipURL: URL?
    private let writingQueue = DispatchQueue(label: "criclips.video.write", qos: .userInitiated)

    private var pixelBufferPool: CVPixelBufferPool?
    private var videoWidth = 1920
    private var videoHeight = 1080
    private var pixelFormat: OSType = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange

    private var preTriggerBuffer: [BufferedFrame] = []
    private var nextPTS: CMTime = .zero
    private var writtenFrameCount = 0
    private var postFrameCount = 0

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
            guard self.state == .buffering else { return }
            guard self.isReadyForTrigger else { return }
            guard self.preTriggerBuffer.count >= self.preTriggerTarget else { return }

            DispatchQueue.main.async {
                self.isReadyForTrigger = false
                self.isRecording = true
            }
            self.startClip()
        }
    }

    // MARK: - Frame ingestion

    func addFrame(_ sampleBuffer: CMSampleBuffer) {
        writingQueue.async { [weak self] in
            guard let self else { return }
            guard let src = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

            switch self.state {
            case .finishing:
                return

            case .streamingPost:
                // Feed the camera's buffer straight to the encoder — no copy, no buffering.
                // The adaptor retains it only until the hardware encoder consumes it.
                self.appendFrame(src)
                self.postFrameCount += 1
                if self.postFrameCount >= self.postTriggerTarget {
                    self.finishClip()
                }

            case .buffering:
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
                self.preTriggerBuffer.append(BufferedFrame(pixelBuffer: copy, timestamp: ts))
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

    /// Opens the asset writer and drains the pre-trigger ring into the encoder.
    /// Runs on writingQueue.
    private func startClip() {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let name = "criclip_\(fmt.string(from: Date())).mp4"
        let outURL = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(name)

        do {
            let writer = try AVAssetWriter(outputURL: outURL, fileType: .mp4)

            let fps = Int(config.frameRate.value)
            let videoSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: videoWidth,
                AVVideoHeightKey: videoHeight,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: targetBitRate(),
                    AVVideoExpectedSourceFrameRateKey: fps,
                    AVVideoMaxKeyFrameIntervalKey: fps * 2,
                    AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
                ]
            ]

            let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            // Realtime mode: the encoder paces itself to keep isReadyForMoreMediaData true
            // while we stream live post-trigger frames into it.
            input.expectsMediaDataInRealTime = true
            input.transform = .identity   // pixel data is already portrait (videoRotationAngle=90 rotates it)

            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                    kCVPixelBufferWidthKey  as String: videoWidth,
                    kCVPixelBufferHeightKey as String: videoHeight
                ])

            guard writer.canAdd(input) else { failWrite("Cannot add video input"); return }
            writer.add(input)
            guard writer.startWriting() else {
                failWrite(writer.error?.localizedDescription ?? "startWriting failed")
                return
            }
            writer.startSession(atSourceTime: .zero)

            self.assetWriter = writer
            self.videoInput = input
            self.adaptor = adaptor
            self.currentClipURL = outURL
            self.nextPTS = .zero
            self.writtenFrameCount = 0
            self.postFrameCount = 0

            // Drain the pre-trigger ring into the encoder, releasing each frame as we go
            // so its pool buffer is recycled immediately.
            while !preTriggerBuffer.isEmpty {
                let frame = preTriggerBuffer.removeFirst()
                appendFrame(frame.pixelBuffer)
            }

            state = .streamingPost

        } catch {
            failWrite(error.localizedDescription)
        }
    }

    /// Appends one pixel buffer at the next fixed-rate timestamp. Runs on writingQueue.
    @discardableResult
    private func appendFrame(_ pixelBuffer: CVPixelBuffer) -> Bool {
        guard let input = videoInput, let adaptor else { return false }

        // With expectsMediaDataInRealTime the input should stay ready; give the
        // encoder a short window before dropping the frame rather than stalling
        // the queue (and the camera) indefinitely.
        var waitedMs = 0
        while !input.isReadyForMoreMediaData && waitedMs < 100 {
            Thread.sleep(forTimeInterval: 0.001)
            waitedMs += 1
        }
        guard input.isReadyForMoreMediaData else { return false }

        guard adaptor.append(pixelBuffer, withPresentationTime: nextPTS) else { return false }
        nextPTS = CMTimeAdd(nextPTS, config.frameRate.frameDuration)
        writtenFrameCount += 1
        return true
    }

    /// Closes the writer once the post-trigger frame budget is reached. Runs on writingQueue.
    private func finishClip() {
        guard let writer = assetWriter, let input = videoInput else {
            state = .buffering
            beginCooldown()
            return
        }
        state = .finishing
        input.markAsFinished()

        let written = writtenFrameCount
        let outURL = currentClipURL

        writer.finishWriting { [weak self] in
            guard let self else { return }
            let err = writer.error

            self.writingQueue.async {
                self.assetWriter = nil
                self.videoInput = nil
                self.adaptor = nil
                self.currentClipURL = nil
                self.state = .buffering
            }

            if let err {
                print("❌ [CriClips] VideoWriter: \(err)")
                DispatchQueue.main.async { self.lastError = err.localizedDescription }
            } else if let outURL {
                print("✅ [CriClips] Saved \(outURL.lastPathComponent) (\(written) frames)")
                DispatchQueue.main.async {
                    self.lastSavedURL = outURL
                    NotificationCenter.default.post(name: .newClipSaved, object: outURL)
                }
            }
            self.beginCooldown()
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

    /// Runs on writingQueue.
    private func failWrite(_ msg: String) {
        print("❌ [CriClips] VideoWriter: \(msg)")
        if let writer = assetWriter, writer.status == .writing {
            writer.cancelWriting()
        }
        assetWriter = nil
        videoInput = nil
        adaptor = nil
        currentClipURL = nil
        state = .buffering
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

        if CVPixelBufferIsPlanar(src) {
            for plane in 0..<CVPixelBufferGetPlaneCount(src) {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                      let d = CVPixelBufferGetBaseAddressOfPlane(out, plane) else { return nil }
                let height = CVPixelBufferGetHeightOfPlane(src, plane)
                let srcBPR = CVPixelBufferGetBytesPerRowOfPlane(src, plane)
                let dstBPR = CVPixelBufferGetBytesPerRowOfPlane(out, plane)
                if srcBPR == dstBPR {
                    memcpy(d, s, srcBPR * height)
                } else {
                    let rowBytes = min(srcBPR, dstBPR)
                    for row in 0..<height {
                        memcpy(d + row * dstBPR, s + row * srcBPR, rowBytes)
                    }
                }
            }
        } else if let s = CVPixelBufferGetBaseAddress(src),
                  let d = CVPixelBufferGetBaseAddress(out) {
            let height = CVPixelBufferGetHeight(src)
            let srcBPR = CVPixelBufferGetBytesPerRow(src)
            let dstBPR = CVPixelBufferGetBytesPerRow(out)
            if srcBPR == dstBPR {
                memcpy(d, s, srcBPR * height)
            } else {
                let rowBytes = min(srcBPR, dstBPR)
                for row in 0..<height {
                    memcpy(d + row * dstBPR, s + row * srcBPR, rowBytes)
                }
            }
        }
        return out
    }

    // MARK: - Helpers

    /// ~0.2 bits per pixel per frame — generous enough that a 150 km/h ball stays
    /// crisp instead of smearing into macroblocks. 1080p30 ≈ 12 Mbps, 1080p60 ≈ 25 Mbps,
    /// 4K30 ≈ 50 Mbps, 4K60 ≈ 100 Mbps.
    private func targetBitRate() -> Int {
        let bitsPerPixel = 0.2
        let rate = Double(videoWidth * videoHeight) * config.frameRate.value * bitsPerPixel
        return max(8_000_000, Int(rate))
    }

    /// Runs on writingQueue.
    private func cleanup() {
        preTriggerBuffer.removeAll()
        if let writer = assetWriter, writer.status == .writing {
            writer.cancelWriting()
        }
        assetWriter = nil
        videoInput = nil
        adaptor = nil
        currentClipURL = nil
        state = .buffering
        postFrameCount = 0
        writtenFrameCount = 0
        DispatchQueue.main.async { [weak self] in self?.isRecording = false }
    }
}
