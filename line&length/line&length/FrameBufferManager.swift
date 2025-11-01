//
//  FrameBufferManager.swift
//  line&length
//
//  Created by Jean Daniel Browne on 22.10.2025.
//

import AVFoundation

class FrameBufferManager: ObservableObject {
    @Published var bufferSize: Int = 0
    @Published var isRecording = false
    
    private var frameBuffer: [CMSampleBuffer] = []
    private var maxBufferSize: Int = FrameBufferConstants.preTriggerFrames
    private let bufferQueue = DispatchQueue(label: "frame.buffer.queue")
    
    var onRecordingStart: (([CMSampleBuffer]) -> Void)?
    var onRecordingStop: (() -> Void)?
    
    init() {
        bufferSize = maxBufferSize
    }
    
    func addFrame(_ sampleBuffer: CMSampleBuffer) {
        bufferQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Create a copy of the sample buffer to store in buffer
            var newSampleBuffer: CMSampleBuffer?
            let status = CMSampleBufferCreateCopy(allocator: kCFAllocatorDefault,
                                                sampleBuffer: sampleBuffer,
                                                sampleBufferOut: &newSampleBuffer)
            
            guard status == noErr, let copiedBuffer = newSampleBuffer else {
                print("Failed to copy sample buffer")
                return
        }
            
            // Add to circular buffer
            self.frameBuffer.append(copiedBuffer)
            
            // Maintain buffer size
            if self.frameBuffer.count > self.maxBufferSize {
                let removedBuffer = self.frameBuffer.removeFirst()
                CMSampleBufferInvalidate(removedBuffer)
            }
            
            DispatchQueue.main.async {
                self.bufferSize = self.frameBuffer.count
            }
        }
    }
    
    func startRecording() {
        guard !isRecording else { return }
        
        bufferQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Copy pre-trigger frames from buffer
            if self.frameBuffer.count == self.maxBufferSize {
                let recordingFrames = Array(self.frameBuffer)
                
                DispatchQueue.main.async {
                    self.isRecording = true
                    self.onRecordingStart?(recordingFrames)
                }
            } else {
                print("⚠️ Buffer not full for pre-trigger recording: \(self.frameBuffer.count)/\(self.maxBufferSize)")
            }
        }
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.isRecording = false
            self.onRecordingStop?()
        }
    }
    
    func addRecordingFrame(_ sampleBuffer: CMSampleBuffer) {
        guard isRecording else { return }
        
        // Create a copy of the sample buffer
        var newSampleBuffer: CMSampleBuffer?
        let status = CMSampleBufferCreateCopy(allocator: kCFAllocatorDefault,
                                            sampleBuffer: sampleBuffer,
                                            sampleBufferOut: &newSampleBuffer)
        
        guard status == noErr, let copiedBuffer = newSampleBuffer else {
            print("Failed to copy post-trigger sample buffer")
            return
        }
        
        // Send the frame to the video writer through callback
        onRecordingStart?([copiedBuffer])
    }
    
    func clearBuffer() {
        bufferQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Invalidate all buffered sample buffers
            for buffer in self.frameBuffer {
                CMSampleBufferInvalidate(buffer)
            }
            
            self.frameBuffer.removeAll()
            
            DispatchQueue.main.async {
                self.bufferSize = 0
            }
        }
    }
    
    deinit {
        clearBuffer()
    }
}
