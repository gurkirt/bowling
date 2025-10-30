//
//  VideoWriter.swift
//  line&length
//
//  Created by Jean Daniel Browne on 22.10.2025.
//

import AVFoundation

class VideoWriter: ObservableObject {
    @Published var isReadyForTrigger = false
    
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private let writingQueue = DispatchQueue(label: "video.writing.queue")
    
    private let bufferSize =  12 // 3 frames buffer
    private var cyclicBuffer: [CMSampleBuffer] = []
    private var cameraStartTime: Date?
    private var isWriting = false
    
    func startCamera() {
        cameraStartTime = Date()
        cyclicBuffer.removeAll()
        isReadyForTrigger = false
        isWriting = false
        print("🎥 Camera started - filling buffer")
    }
    
    func triggerRecording() {
        guard isReadyForTrigger, cyclicBuffer.count == bufferSize else {
            print("⚠️ Not ready - buffer has \(cyclicBuffer.count)/\(bufferSize) frames")
            return
        }
        
        print("🎬 Trigger pressed - saving \(cyclicBuffer.count) frames")
        writingQueue.async { [weak self] in
            self?.writeBufferedFrames()
        }
    }
    
    func addFrame(_ sampleBuffer: CMSampleBuffer) {
        writingQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Don't add frames while writing to prevent race conditions
            if self.isWriting {
                print("⏸️ Skipping frame - currently writing")
                return
            }
            
            // Create a copy to ensure the buffer is retained
            guard let bufferCopy = createCMSampleBufferCopy(sampleBuffer) else {
                print("⚠️ Failed to create buffer copy")
                return
            }
            
            let frameIndex = self.cyclicBuffer.count
            self.cyclicBuffer.append(bufferCopy)
            print("📦 Buffer count: \(self.cyclicBuffer.count)/\(self.bufferSize), adding frame at index: \(frameIndex)")
            
            // Keep only last 15 frames
            if self.cyclicBuffer.count > self.bufferSize {
                self.cyclicBuffer.removeFirst()
                print("🗑️ Removed oldest frame, now starts at index 0")
            }
            
            // Set ready when buffer is full
            if self.cyclicBuffer.count == self.bufferSize {
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
    
    private func writeBufferedFrames() {
        guard cyclicBuffer.count == bufferSize else { return }
        
        // Mark as writing to prevent buffer modifications
        isWriting = true
        
        // Take a snapshot of the buffer to avoid race conditions
        let buffersToWrite = cyclicBuffer
        
        print("🎬 Writing snapshot of \(buffersToWrite.count) frames")
        
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
                AVVideoWidthKey: 1920,
                AVVideoHeightKey: 1080,
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
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: 1920,
                    kCVPixelBufferHeightKey as String: 1080
                ]
            )
            
            guard assetWriter!.canAdd(videoInput!) else {
                print("❌ Cannot add video input")
                return
            }
            assetWriter!.add(videoInput!)
            
            // Write frames
            guard assetWriter!.startWriting() else {
                print("❌ Cannot start writing")
                return
            }
            
            assetWriter!.startSession(atSourceTime: .zero)
            
            let timePerFrame = CMTime(seconds: 1.0/30.0, preferredTimescale: 600)
            var currentTime = CMTime.zero
            var framesWritten = 0
            
            print("📝 Writing \(buffersToWrite.count) frames from buffer")
            
            // Write all frames synchronously since expectsMediaDataInRealTime = false
            for (index, sampleBuffer) in buffersToWrite.enumerated() {
                // Wait for input to be ready
                while !videoInput!.isReadyForMoreMediaData {
                    Thread.sleep(forTimeInterval: 0.001) // Wait 1ms
                }
                
                print("🔍 Extracting pixel buffer for frame \(index)...")
                guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                    print("❌ Failed to get pixel buffer for frame \(index)")
                    print("   Sample buffer valid: \(sampleBuffer)")
                    print("   Has attachments: \(CMSampleBufferGetNumSamples(sampleBuffer)) samples")
                    continue
                }
                
                let width = CVPixelBufferGetWidth(pixelBuffer)
                let height = CVPixelBufferGetHeight(pixelBuffer)
                print("   Got pixel buffer: \(width)x\(height)")
                
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
            
            // Wait for writer to finish asynchronously
            let finishGroup = DispatchGroup()
            finishGroup.enter()
            
            assetWriter!.finishWriting { [framesWritten] in
                if let error = self.assetWriter?.error {
                    print("❌ Video writer error: \(error)")
                } else {
                    print("✅ Video saved: \(videoURL.path) with \(framesWritten) frames")
                }
                self.isWriting = false
                finishGroup.leave()
            }
            
            finishGroup.wait()
            
        } catch {
            print("❌ Error: \(error)")
            isWriting = false
        }
    }
    
    // Helper function to create a copy of CMSampleBuffer
    private func createCMSampleBufferCopy(_ sampleBuffer: CMSampleBuffer) -> CMSampleBuffer? {
        var bufferCopy: CMSampleBuffer?
        let status = CMSampleBufferCreateCopy(allocator: kCFAllocatorDefault, sampleBuffer: sampleBuffer, sampleBufferOut: &bufferCopy)
        if status != noErr {
            print("❌ Failed to create sample buffer copy: \(status)")
            return nil
        }
        return bufferCopy
    }
}

// Helper extension
private extension DateFormatter {
    func apply(_ closure: (DateFormatter) -> Void) -> DateFormatter {
        closure(self)
        return self
    }
}
