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
    
    weak var videoWriter: VideoWriter?
    private var internalFrameCount = 0
    private var internalDroppedCount = 0
    private var configuration = RecordingConfiguration.default
    
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
            #if DEBUG
            print(logMessage) // Console spam only in DEBUG builds
            #endif
        }
    }
    
    func checkAuthorization() {
        addDebugMessage("Checking camera authorization...")
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            addDebugMessage("Camera authorization: GRANTED")
            isAuthorized = true
            DispatchQueue.main.async {
                self.videoWriter?.startCamera() // Initialize VideoWriter first
            }
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
            self.applySessionPreset()
            
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
                if #available(iOS 17.0, *) {
                    if connection.isVideoRotationAngleSupported(90) {
                        connection.videoRotationAngle = 90  // 90 degrees for portrait
                    }
                } else {
                    if connection.isVideoOrientationSupported {
                        connection.videoOrientation = .portrait
                    }
                }
            }
            
            if self.captureSession.canAddOutput(self.videoOutput) {
                self.captureSession.addOutput(self.videoOutput)
                // self.addDebugMessage("Video output added successfully")
            } else {
                self.addDebugMessage("ERROR: Cannot add video output to session")
            }
            
            self.applyFormatAndFrameRate(to: camera)
            
            self.captureSession.commitConfiguration()
            // self.addDebugMessage("Capture session configuration committed")
            
            // Start the session on the session queue (background thread)
            // self.addDebugMessage("Starting capture session...")
            self.captureSession.startRunning()
            self.addDebugMessage("Session running: \(self.captureSession.isRunning)")
            
            // Update UI state on main thread
            DispatchQueue.main.async {
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

    func updateConfiguration(_ configuration: RecordingConfiguration) {
        addDebugMessage("Updating configuration: \(configuration.resolution.displayName) @ \(Int(configuration.frameRate.value))fps")
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.configuration = configuration
            guard let input = self.captureSession.inputs.compactMap({ $0 as? AVCaptureDeviceInput }).first else {
                self.addDebugMessage("Capture session not ready - configuration stored for later")
                return
            }
            self.captureSession.beginConfiguration()
            self.applySessionPreset()
            self.applyFormatAndFrameRate(to: input.device)
            self.captureSession.commitConfiguration()
        }
    }
    
    private func applySessionPreset() {
        let desiredPreset = configuration.resolution.sessionPreset
        if captureSession.canSetSessionPreset(desiredPreset) {
            captureSession.sessionPreset = desiredPreset
            addDebugMessage("Session preset set to \(configuration.resolution.displayName)")
        } else if captureSession.canSetSessionPreset(.high) {
            captureSession.sessionPreset = .high
            addDebugMessage("Session preset fallback to HIGH")
        } else if captureSession.canSetSessionPreset(.medium) {
            captureSession.sessionPreset = .medium
            addDebugMessage("Session preset fallback to MEDIUM")
        } else {
            addDebugMessage("WARNING: Unable to set session preset")
        }
    }

    private func applyFormatAndFrameRate(to device: AVCaptureDevice) {
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            if let format = bestMatchingFormat(for: configuration, device: device) {
                if device.activeFormat != format {
                    let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                    addDebugMessage("Active format set to \(dims.width)x\(dims.height)")
                    device.activeFormat = format
                }
            } else {
                addDebugMessage("WARNING: No exact format for \(configuration.resolution.displayName) @ \(Int(configuration.frameRate.value))fps")
            }
            let resolvedRate = resolvedFrameRate(desired: configuration.frameRate.value, format: device.activeFormat)
            let duration = CMTime(seconds: 1.0 / resolvedRate, preferredTimescale: 600)
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            let formattedRate = String(format: "%.2f", resolvedRate)
            addDebugMessage("Frame rate set to \(formattedRate) fps")
        } catch {
            addDebugMessage("ERROR: Failed to apply format/frame rate: \(error.localizedDescription)")
        }
    }

    private func bestMatchingFormat(for configuration: RecordingConfiguration, device: AVCaptureDevice) -> AVCaptureDevice.Format? {
        let desiredDimensions = configuration.resolution.dimensions
        let desiredFrameRate = configuration.frameRate.value
        let tolerance = 0.5
        let formats = device.formats
        guard !formats.isEmpty else { return nil }
        let sorted = formats.sorted { lhs, rhs in
            let lhsPenalty = formatPenalty(for: lhs, desired: desiredDimensions)
            let rhsPenalty = formatPenalty(for: rhs, desired: desiredDimensions)
            if lhsPenalty == rhsPenalty {
                let lhsMax = lhs.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
                let rhsMax = rhs.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
                if abs(lhsMax - rhsMax) < 0.1 {
                    let lhsDims = CMVideoFormatDescriptionGetDimensions(lhs.formatDescription)
                    let rhsDims = CMVideoFormatDescriptionGetDimensions(rhs.formatDescription)
                    if lhsDims.width == rhsDims.width {
                        return lhsDims.height > rhsDims.height
                    }
                    return lhsDims.width > rhsDims.width
                }
                return lhsMax > rhsMax
            }
            return lhsPenalty < rhsPenalty
        }
        let supporting = sorted.first { format in
            format.videoSupportedFrameRateRanges.contains { range in
                desiredFrameRate >= range.minFrameRate - tolerance && desiredFrameRate <= range.maxFrameRate + tolerance
            }
        }
        return supporting ?? sorted.first
    }

    private func formatPenalty(for format: AVCaptureDevice.Format, desired: (width: Int, height: Int)) -> Int {
        let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let width = Int(dimensions.width)
        let height = Int(dimensions.height)
        let underWidth = max(0, desired.width - width)
        let underHeight = max(0, desired.height - height)
        let overWidth = max(0, width - desired.width)
        let overHeight = max(0, height - desired.height)
        let undersizePenalty = (underWidth + underHeight) * 10_000
        return undersizePenalty + overWidth + overHeight
    }

    private func resolvedFrameRate(desired: Double, format: AVCaptureDevice.Format) -> Double {
        let ranges = format.videoSupportedFrameRateRanges
        guard !ranges.isEmpty else { return desired }
        let tolerance = 0.1
        if let matching = ranges.first(where: { desired >= $0.minFrameRate - tolerance && desired <= $0.maxFrameRate + tolerance }) {
            return min(max(desired, matching.minFrameRate), matching.maxFrameRate)
        }
        let closest = ranges.min { lhs, rhs in
            let lhsDelta = abs(lhs.maxFrameRate - desired)
            let rhsDelta = abs(rhs.maxFrameRate - desired)
            return lhsDelta < rhsDelta
        }
        guard let range = closest else { return desired }
        if desired < range.minFrameRate {
            return range.minFrameRate
        }
        return range.maxFrameRate
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
