//
//  VideoWriter.swift
//  line&length
//
//  Created by Jean Daniel Browne on 22.10.2025.
//

import AVFoundation
import CoreVideo

// Shared constants for frame buffering
enum FrameBufferConstants {
    /// Pre-trigger frames to buffer (1 second at 30fps)
    static let preTriggerFrames = 30
    /// Post-trigger frames to record (2 seconds at 30fps)
    static let postTriggerFrames = 60
    /// Total buffer size including safety margin
    static let totalBufferSize = preTriggerFrames + postTriggerFrames + 2
}

// Structure to hold copied pixel buffer and its timestamp
private struct BufferedFrame {
    let pixelBuffer: CVPixelBuffer
    let timestamp: CMTime
}

class VideoWriter: ObservableObject {
    @Published var isReadyForTrigger = false
    @Published var lastError: String?
    
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private let writingQueue = DispatchQueue(label: "video.writing.queue")
    
    // Custom pixel buffer pool to avoid fighting with AVFoundation's internal pool
    private var pixelBufferPool: CVPixelBufferPool?
    private var poolAttributes: [String: Any] = [:]
    private var pixelBufferAttributes: [String: Any] = [:]
    
    private let preTriggerSize = FrameBufferConstants.preTriggerFrames
    private var preTriggerBuffer: [BufferedFrame] = []
    private var postTriggerBuffer: [BufferedFrame] = []
    private var cameraStartTime: Date?
    private var isWriting = false
    private var isCollectingPostTrigger = false
    private var postTriggerCount = 0
    
    // Video dimensions - will be set from first frame
    private var videoWidth: Int = 1920
    private var videoHeight: Int = 1080
    private var pixelFormat: OSType = kCVPixelFormatType_32BGRA
    
    func startCamera() {
        cleanup()
        cameraStartTime = Date()
        isReadyForTrigger = false
        print("🎥 Camera started - filling buffer")
    }
    
    func triggerRecording() {
        guard !isWriting else {
            print("⚠️ Cannot trigger - currently writing previous recording")
            return
        }
        
        guard !isCollectingPostTrigger else {
            print("⚠️ Cannot trigger - already collecting post-trigger frames")
            return
        }
        
        guard isReadyForTrigger, preTriggerBuffer.count == preTriggerSize else {
            print("⚠️ Not ready - buffer has \(preTriggerBuffer.count)/\(preTriggerSize) frames")
            return
        }
        
        print("🎬 Trigger pressed - collecting post-trigger frames")
        writingQueue.async { [weak self] in
            guard let self = self else { return }
            self.isCollectingPostTrigger = true
            self.postTriggerBuffer.removeAll()
            self.postTriggerCount = 0
            print("✅ Starting post-trigger collection phase")
        }
    }
    
    func addFrame(_ sampleBuffer: CMSampleBuffer) {
        writingQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Don't add frames while writing final video
            if self.isWriting {
                print("⏸️ Skipping frame - currently writing")
                return
            }
            
            // Extract pixel buffer and timestamp from sample buffer
            guard let sourcePixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                print("⚠️ Failed to get pixel buffer from sample buffer")
                return
            }
            
            let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            
            // Initialize pool dimensions from first frame if needed
            if self.pixelBufferPool == nil {
                let width = CVPixelBufferGetWidth(sourcePixelBuffer)
                let height = CVPixelBufferGetHeight(sourcePixelBuffer)
                let format = CVPixelBufferGetPixelFormatType(sourcePixelBuffer)
                
                self.videoWidth = width
                self.videoHeight = height
                self.pixelFormat = format
                
                if !self.createPixelBufferPool(width: width, height: height, format: format) {
                    print("❌ Failed to create pixel buffer pool")
                    return
                }
                print("✅ Created pixel buffer pool: \(width)x\(height), format: \(format)")
            }
            
            // Create a copy using our custom pool and manual memcpy
            guard let copiedBuffer = self.copyPixelBuffer(sourcePixelBuffer) else {
                print("⚠️ Failed to copy pixel buffer")
                return
            }
            
            let bufferedFrame = BufferedFrame(pixelBuffer: copiedBuffer, timestamp: timestamp)
            
            if self.isCollectingPostTrigger {
                // Handle post-trigger frame collection
                self.postTriggerBuffer.append(bufferedFrame)
                self.postTriggerCount += 1
                print("📦 Post-trigger frame \(self.postTriggerCount)/\(FrameBufferConstants.postTriggerFrames)")
                
                // If we have all post-trigger frames, start writing complete video
                if self.postTriggerCount >= FrameBufferConstants.postTriggerFrames {
                    self.isCollectingPostTrigger = false
                    self.writeCompleteRecording()
                }
            } else {
                // Handle pre-trigger buffer
                self.preTriggerBuffer.append(bufferedFrame)
                print("📦 Pre-trigger buffer: \(self.preTriggerBuffer.count)/\(self.preTriggerSize)")
                
                // Keep only required pre-trigger frames
                if self.preTriggerBuffer.count > self.preTriggerSize {
                    let removedFrame = self.preTriggerBuffer.removeFirst()
                    CVPixelBufferUnlockBaseAddress(removedFrame.pixelBuffer, .readOnly)
                    print("🗑️ Removed oldest pre-trigger frame")
                }
                
                // Set ready when pre-trigger buffer is full
                if self.preTriggerBuffer.count == self.preTriggerSize {
                    if let startTime = self.cameraStartTime {
                        let elapsed = Date().timeIntervalSince(startTime)
                        if elapsed >= 0.5 {  // 0.5 seconds minimum
                            if !self.isReadyForTrigger {
                                DispatchQueue.main.async {
                                    self.isReadyForTrigger = true
                                    print("✅ Ready for trigger")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    private func writeCompleteRecording() {
        // Mark as writing to prevent buffer modifications
        isWriting = true
        
        // Take snapshots of buffers to prevent race conditions
        let preBufferSnapshot = Array(preTriggerBuffer)
        let postBufferSnapshot = Array(postTriggerBuffer)
        
        // Combine pre and post trigger frames
        let buffersToWrite = preBufferSnapshot + postBufferSnapshot
        print("🎬 Writing complete recording: \(buffersToWrite.count) frames (\(preTriggerBuffer.count) pre + \(postTriggerBuffer.count) post)")
        
        // Create filename
        let timestamp = DateFormatter().apply {
            $0.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        }.string(from: Date())
        
        let fileName = "trigger_\(timestamp).mp4"
        let videoURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(fileName)
        
        // Setup writer
        do {
            assetWriter = try AVAssetWriter(outputURL: videoURL, fileType: .mp4)
            
            let videoSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: videoWidth,
                AVVideoHeightKey: videoHeight,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: 5_000_000
                ]
            ]
            
            videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            videoInput?.expectsMediaDataInRealTime = false  
            videoInput?.transform = CGAffineTransform(rotationAngle: CGFloat.pi / 2)
            
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: videoInput!,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                    kCVPixelBufferWidthKey as String: videoWidth,
                    kCVPixelBufferHeightKey as String: videoHeight
                ]
            )
            self.adaptor = adaptor
            
            guard assetWriter!.canAdd(videoInput!) else {
                print("❌ Cannot add video input")
                isWriting = false
                return
            }
            assetWriter!.add(videoInput!)
            
            // Write frames
            guard assetWriter!.startWriting() else {
                print("❌ Cannot start writing: \(assetWriter?.error?.localizedDescription ?? "unknown error")")
                isWriting = false
                return
            }
            
            assetWriter!.startSession(atSourceTime: .zero)
            
            let timePerFrame = CMTime(seconds: 1.0/30.0, preferredTimescale: 600)
            var currentTime = CMTime.zero
            var framesWritten = 0
            
            print("📝 Writing \(buffersToWrite.count) frames from buffer")
            
            // Write all frames synchronously since expectsMediaDataInRealTime = false
            for (index, bufferedFrame) in buffersToWrite.enumerated() {
                // Wait for input to be ready
                while !videoInput!.isReadyForMoreMediaData {
                    Thread.sleep(forTimeInterval: 0.001) // Wait 1ms
                }
                
                let pixelBuffer = bufferedFrame.pixelBuffer
                let width = CVPixelBufferGetWidth(pixelBuffer)
                let height = CVPixelBufferGetHeight(pixelBuffer)
                
                print("🔍 Writing frame \(index): \(width)x\(height)")
                
                let success = adaptor.append(pixelBuffer, withPresentationTime: currentTime)
                if success {
                    framesWritten += 1
                    currentTime = CMTimeAdd(currentTime, timePerFrame)
                    print("✅ Appended frame \(framesWritten)/\(buffersToWrite.count)")
                } else {
                    print("⚠️ adaptor.append() failed for frame \(index)")
                }
            }
            
            print("📝 All \(framesWritten) frames appended")
            
            videoInput!.markAsFinished()
            
            // Finish writing asynchronously - don't block the queue
            assetWriter!.finishWriting { [weak self, framesWritten] in
                guard let self = self else { return }
                
                DispatchQueue.main.async {
                    if let error = self.assetWriter?.error {
                        print("❌ Video writer error: \(error)")
                    } else {
                        print("✅ Video saved: \(videoURL.path) with \(framesWritten) frames")
                    }
                    
                    self.isWriting = false
                    self.isReadyForTrigger = true // Reset ready state for next recording
                }
            }
            
        } catch {
            let errorMsg = "❌ Error writing video: \(error.localizedDescription)"
            print(errorMsg)
            isWriting = false
            DispatchQueue.main.async { [weak self] in
                self?.lastError = errorMsg
                self?.isReadyForTrigger = true
            }
        }
    }
    
    // MARK: - Pixel Buffer Pool Management
    
    /// Creates a custom CVPixelBufferPool to avoid fighting with AVFoundation's internal pool
    private func createPixelBufferPool(width: Int, height: Int, format: OSType) -> Bool {
        // Pool attributes - specify minimum buffer count
        poolAttributes = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: FrameBufferConstants.totalBufferSize
        ]
        
        // Pixel buffer attributes
        pixelBufferAttributes = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        
        let status = CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            poolAttributes as CFDictionary,
            pixelBufferAttributes as CFDictionary,
            &pixelBufferPool
        )
        
        if status != kCVReturnSuccess {
            print("❌ Failed to create pixel buffer pool: \(status)")
            return false
        }
        
        return true
    }
    
    private func cleanup() {
        // Clear pre-trigger buffer and ensure pixel buffers are released
        for frame in preTriggerBuffer {
            CVPixelBufferUnlockBaseAddress(frame.pixelBuffer, .readOnly)
        }
        preTriggerBuffer.removeAll()
        
        // Clear post-trigger buffer and ensure pixel buffers are released
        for frame in postTriggerBuffer {
            CVPixelBufferUnlockBaseAddress(frame.pixelBuffer, .readOnly)
        }
        postTriggerBuffer.removeAll()
        
        // Reset state
        isCollectingPostTrigger = false
        postTriggerCount = 0
        isWriting = false
        
        // Clear pixel buffer pool
        pixelBufferPool = nil
    }
    
    deinit {
        cleanup()
    }
    
    /// Copies pixel buffer using manual memcpy from our custom pool
    private func copyPixelBuffer(_ sourceBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let pool = pixelBufferPool else {
            print("❌ Pixel buffer pool not initialized")
            return nil
        }
        
        // Create a new pixel buffer from our custom pool
        var destinationBuffer: CVPixelBuffer?
        let poolStatus = CVPixelBufferPoolCreatePixelBuffer(
            kCFAllocatorDefault,
            pool,
            &destinationBuffer
        )
        
        guard poolStatus == kCVReturnSuccess, let destBuffer = destinationBuffer else {
            print("❌ Failed to create pixel buffer from pool: \(poolStatus)")
            return nil
        }
        
        // Get dimensions
        let width = CVPixelBufferGetWidth(sourceBuffer)
        let height = CVPixelBufferGetHeight(sourceBuffer)
        let sourceFormat = CVPixelBufferGetPixelFormatType(sourceBuffer)
        let destFormat = CVPixelBufferGetPixelFormatType(destBuffer)
        
        guard sourceFormat == destFormat else {
            print("❌ Format mismatch: source=\(sourceFormat), dest=\(destFormat)")
            return nil
        }
        
        // Lock both buffers for direct memory access
        let lockFlags = CVPixelBufferLockFlags(rawValue: 0)
        
        guard CVPixelBufferLockBaseAddress(sourceBuffer, lockFlags) == kCVReturnSuccess else {
            print("❌ Failed to lock source buffer")
            return nil
        }
        
        defer {
            CVPixelBufferUnlockBaseAddress(sourceBuffer, lockFlags)
        }
        
        guard CVPixelBufferLockBaseAddress(destBuffer, lockFlags) == kCVReturnSuccess else {
            print("❌ Failed to lock destination buffer")
            return nil
        }
        
        // Get base addresses and bytes per row
        let sourceBaseAddress = CVPixelBufferGetBaseAddress(sourceBuffer)
        let destBaseAddress = CVPixelBufferGetBaseAddress(destBuffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(sourceBuffer)
        let destBytesPerRow = CVPixelBufferGetBytesPerRow(destBuffer)
        
        guard let srcAddr = sourceBaseAddress, let dstAddr = destBaseAddress else {
            CVPixelBufferUnlockBaseAddress(destBuffer, lockFlags)
            print("❌ Failed to get base addresses")
            return nil
        }
        
        // Calculate bytes per pixel (for 32BGRA it's 4 bytes)
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        
        // Copy row by row to handle potential row alignment differences
        let minBytesPerRow = min(sourceBytesPerRow, destBytesPerRow)
        let copyBytesPerRow = min(minBytesPerRow, bytesPerRow)
        
        for row in 0..<height {
            let sourceRow = srcAddr.advanced(by: row * sourceBytesPerRow)
            let destRow = dstAddr.advanced(by: row * destBytesPerRow)
            memcpy(destRow, sourceRow, copyBytesPerRow)
        }
        
        // Unlock destination buffer
        CVPixelBufferUnlockBaseAddress(destBuffer, lockFlags)
        
        return destBuffer
    }
}



private extension DateFormatter {
    func apply(_ closure: (DateFormatter) -> Void) -> DateFormatter {
        closure(self)
        return self
    }
}
