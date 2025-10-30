//
//  CameraManager.swift
//  line&length
//
//  Created by Jean Daniel Browne on 22.10.2025.
//

import AVFoundation
import Combine

class CameraManager: NSObject, ObservableObject {
    @Published var isAuthorized = false
    @Published var isSessionRunning = false
    @Published var error: CameraError?
    @Published var debugMessages: [String] = []
    @Published var frameCount = 0
    @Published var droppedFrameCount = 0
    private var internalFrameCount = 0
    private var internalDroppedCount = 0
    
    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    
    var frameHandler: ((CMSampleBuffer) -> Void)?
    
    override init() {
        super.init()
        addDebugMessage("CameraManager initialized")
        checkAuthorization()
    }
    
    private func addDebugMessage(_ message: String) {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        let timestamp = formatter.string(from: Date())
        let logMessage = "[\(timestamp)] \(message)"
        DispatchQueue.main.async {
            self.debugMessages.append(logMessage)
            print(logMessage) // Also print to console
        }
    }
    
    func checkAuthorization() {
        addDebugMessage("Checking camera authorization...")
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            addDebugMessage("Camera authorization: GRANTED")
            isAuthorized = true
            setupCaptureSession()
        case .notDetermined:
            addDebugMessage("Camera authorization: NOT DETERMINED - requesting access")
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.addDebugMessage("Camera authorization result: \(granted ? "GRANTED" : "DENIED")")
                    self?.isAuthorized = granted
                    if granted {
                        self?.setupCaptureSession()
                    }
                }
            }
        case .denied, .restricted:
            addDebugMessage("Camera authorization: DENIED/RESTRICTED")
            isAuthorized = false
            error = .authorizationDenied
        @unknown default:
            addDebugMessage("Camera authorization: UNKNOWN STATUS")
            isAuthorized = false
            error = .unknown
        }
    }
    
    private func setupCaptureSession() {
        addDebugMessage("Starting camera setup on background queue...")
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.addDebugMessage("Beginning capture session configuration...")
            self.captureSession.beginConfiguration()
            
            // Configure session preset for optimal performance (use high for better quality)
            if self.captureSession.canSetSessionPreset(.high) {
                self.captureSession.sessionPreset = .high
                self.addDebugMessage("Session preset set to HIGH for better quality")
            } else if self.captureSession.canSetSessionPreset(.medium) {
                self.captureSession.sessionPreset = .medium
                self.addDebugMessage("Session preset set to MEDIUM")
            } else {
                self.addDebugMessage("WARNING: Cannot set session preset")
            }
            
            // Setup camera input (back camera)
            guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                self.addDebugMessage("ERROR: No back camera found")
                 DispatchQueue.main.async {
                    self.error = .noCamera
                }
                return
            }
            
            self.addDebugMessage("Back camera found: \(camera.localizedName)")
            
            do {
                let cameraInput = try AVCaptureDeviceInput(device: camera)
                if self.captureSession.canAddInput(cameraInput) {
                    self.captureSession.addInput(cameraInput)
                    self.addDebugMessage("Camera input added successfully")
                } else {
                    self.addDebugMessage("ERROR: Cannot add camera input to session")
                }
            } catch {
                self.addDebugMessage("ERROR: Failed to create camera input: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.error = .setupFailed(error)
                }
                return
            }
            
            // Setup video output with optimized settings
            self.addDebugMessage("Setting up video output...")
            self.videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.output.queue"))
            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            
            // Optimize video output to reduce frame drops
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            
            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            
            if self.captureSession.canAddOutput(self.videoOutput) {
                self.captureSession.addOutput(self.videoOutput)
                self.addDebugMessage("Video output added successfully")
            } else {
                self.addDebugMessage("ERROR: Cannot add video output to session")
            }
            
            self.captureSession.commitConfiguration()
            self.addDebugMessage("Capture session configuration committed")
            
            DispatchQueue.main.async {
                self.addDebugMessage("Starting capture session...")
                self.captureSession.startRunning()
                self.addDebugMessage("Session running: \(self.captureSession.isRunning)")
                self.isSessionRunning = self.captureSession.isRunning
            }
        }
    }
    
    func startSession() {
        addDebugMessage("Manual session start requested")
        sessionQueue.async { [weak self] in
            guard let self = self, !self.captureSession.isRunning else { 
                self?.addDebugMessage("Session already running or self is nil")
                return 
            }
            self.captureSession.startRunning()
            self.addDebugMessage("Manual session start: \(self.captureSession.isRunning)")
            DispatchQueue.main.async {
                self.isSessionRunning = self.captureSession.isRunning
            }
        }
    }
    
    func stopSession() {
        addDebugMessage("Manual session stop requested")
        sessionQueue.async { [weak self] in
            guard let self = self, self.captureSession.isRunning else { 
                self?.addDebugMessage("Session already stopped or self is nil")
                return 
            }
            self.captureSession.stopRunning()
            self.addDebugMessage("Manual session stop: \(self.captureSession.isRunning)")
            DispatchQueue.main.async {
                self.isSessionRunning = self.captureSession.isRunning
            }
        }
    }
    
    func setFrameHandler(_ handler: @escaping (CMSampleBuffer) -> Void) {
        frameHandler = handler
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        internalFrameCount += 1
        
        if internalFrameCount % 10 == 0 {
            DispatchQueue.main.async {
                self.frameCount = self.internalFrameCount
            }
        }
        
        if frameHandler == nil {
            print("⚠️ WARNING: frameHandler is nil at frame \(internalFrameCount)")
        }
        
        frameHandler?(sampleBuffer)
        
        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
        if internalFrameCount % 30 == 0 {
            print("⏱️ captureOutput execution time: \(String(format: "%.4f", elapsedTime * 1000))ms")
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        internalDroppedCount += 1
        print("⚠️ Dropped frame \(internalDroppedCount)")
        
        if internalDroppedCount % 1 == 0 {
            DispatchQueue.main.async {
                self.droppedFrameCount = self.internalDroppedCount
            }
        }
    }
}

enum CameraError: Error, LocalizedError {
    case authorizationDenied
    case noCamera
    case setupFailed(Error)
    case unknown
    
    var errorDescription: String? {
        switch self {
        case .authorizationDenied:
            return "Camera access denied. Please enable camera access in Settings."
        case .noCamera:
            return "No camera available on this device."
        case .setupFailed(let error):
            return "Camera setup failed: \(error.localizedDescription)"
        case .unknown:
            return "An unknown error occurred."
        }
    }
}
