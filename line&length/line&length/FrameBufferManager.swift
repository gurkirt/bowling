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
    
    // Circular buffer implementation
    private var frameBuffer: [CMSampleBuffer?]  // Using optional to explicitly handle nil cases
    private var writeIndex: Int = 0             // Next position to write
    private var readIndex: Int = 0              // Next position to read
    private var currentBufferCount: Int = 0     // Current number of valid frames
    private let maxBufferSize: Int = FrameBufferConstants.preTriggerFrames
    private let bufferQueue = DispatchQueue(label: "frame.buffer.queue")
    
    var onRecordingStart: (([CMSampleBuffer]) -> Void)?
    var onRecordingStop: (() -> Void)?
    
    init() {
        // Pre-allocate buffer with exact size to avoid resizing
        frameBuffer = Array(repeating: nil, count: maxBufferSize)
        bufferSize = 0
    }
    
    func addFrame(_ sampleBuffer: CMSampleBuffer) {
        bufferQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Only copy if we need this frame (buffer not full or in recording state)
            if self.currentBufferCount < self.maxBufferSize || self.isRecording {
                // If there's an existing buffer, release it properly
                if let oldBuffer = self.frameBuffer[self.writeIndex] {
                    self.frameBuffer[self.writeIndex] = nil
                    CMSampleBufferInvalidate(oldBuffer)
                }
                
                // Create a proper copy of the buffer
                var copy: CMSampleBuffer?
                CMSampleBufferCreateCopy(allocator: kCFAllocatorDefault,
                                       sampleBuffer: sampleBuffer,
                                       sampleBufferOut: &copy)
                
                if let copy = copy {
                    self.frameBuffer[self.writeIndex] = copy
                }
                
                // Update write position
                self.writeIndex = (self.writeIndex + 1) % self.maxBufferSize
                
                // Update buffer count
                if self.currentBufferCount < self.maxBufferSize {
                    self.currentBufferCount += 1
                } else {
                    // When buffer is full, move read index as well
                    self.readIndex = (self.readIndex + 1) % self.maxBufferSize
                }
                
                // Update UI less frequently to reduce overhead
                if self.currentBufferCount % 5 == 0 {
                    DispatchQueue.main.async {
                        self.bufferSize = self.currentBufferCount
                    }
                }
            }
        }
    }
    
    func startRecording() {
        guard !isRecording else { return }
        
        bufferQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Only start if buffer is full
            guard self.currentBufferCount == self.maxBufferSize else {
                return
            }
            
            // Use direct buffer reference instead of copying
            var recordingFrames: [CMSampleBuffer] = []
            recordingFrames.reserveCapacity(self.maxBufferSize)
            
            var index = self.readIndex
            for _ in 0..<self.maxBufferSize {
                if let buffer = self.frameBuffer[index] {
                    // Create a proper copy of the buffer
                    var copy: CMSampleBuffer?
                    CMSampleBufferCreateCopy(allocator: kCFAllocatorDefault,
                                           sampleBuffer: buffer,
                                           sampleBufferOut: &copy)
                    if let copy = copy {
                        recordingFrames.append(copy)
                    }
                }
                index = (index + 1) % self.maxBufferSize
            }
            
            if !recordingFrames.isEmpty {
                DispatchQueue.main.async {
                    self.isRecording = true
                    self.onRecordingStart?(recordingFrames)
                }
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
        
        // Create a proper copy
        var copy: CMSampleBuffer?
        CMSampleBufferCreateCopy(allocator: kCFAllocatorDefault,
                               sampleBuffer: sampleBuffer,
                               sampleBufferOut: &copy)
        
        if let copy = copy {
            // Send the frame directly
            onRecordingStart?([copy])
        }
    }
    
    func clearBuffer() {
        bufferQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Clear and invalidate buffers in one pass
            self.frameBuffer.enumerated().forEach { (i, buffer) in
                if let buffer = buffer {
                    CMSampleBufferInvalidate(buffer)
                }
                self.frameBuffer[i] = nil
            }
            
            // Reset state
            self.writeIndex = 0
            self.readIndex = 0
            self.currentBufferCount = 0
            self.bufferSize = 0  // Update UI directly since we're cleaning up
        }
    }
    
    deinit {
        clearBuffer()
    }
}
