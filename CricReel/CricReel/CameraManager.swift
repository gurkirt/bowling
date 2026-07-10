//
//  CameraManager.swift
//  CriClips
//
//  Configures the back wide-angle camera via AVCaptureSession (1080p/4K, 30/60 fps),
//  requests permission, and streams 4:2:0 YUV frames through a frameHandler callback.
//

import AVFoundation
import Combine

enum CameraError: Error, LocalizedError {
    case authorizationDenied
    case noCamera
    case setupFailed(Error)
    case unknown

    var errorDescription: String? {
        switch self {
        case .authorizationDenied: return "Camera access was denied. Enable it in Settings."
        case .noCamera:            return "No back camera found."
        case .setupFailed(let e):  return "Camera setup failed: \(e.localizedDescription)"
        case .unknown:             return "An unknown camera error occurred."
        }
    }
}

class CameraManager: NSObject, ObservableObject {
    @Published var isAuthorized = false
    @Published var isSessionRunning = false
    @Published var error: CameraError?
    @Published var frameCount = 0
    @Published var droppedFrameCount = 0

    weak var videoWriter: VideoWriter?

    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "criclips.camera.session", qos: .userInteractive)
    private let outputQueue  = DispatchQueue(label: "criclips.camera.output",  qos: .userInteractive)

    private var internalFrameCount   = 0
    private var internalDroppedCount = 0
    private var configuration = RecordingConfiguration.default

    var frameHandler: ((CMSampleBuffer) -> Void)?

    override init() {
        super.init()
        checkAuthorization()
    }

    // MARK: - Authorization

    func checkAuthorization() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isAuthorized = true
            setupCaptureSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.isAuthorized = granted
                    if granted { self?.setupCaptureSession() }
                }
            }
        case .denied, .restricted:
            isAuthorized = false
            error = .authorizationDenied
        @unknown default:
            isAuthorized = false
            error = .unknown
        }
    }

    // MARK: - Session Setup

    private func setupCaptureSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }

            guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                DispatchQueue.main.async { self.error = .noCamera }
                return
            }

            self.captureSession.beginConfiguration()

            // .inputPriority hands full format control to us;
            // the session will never override camera.activeFormat.
            self.captureSession.sessionPreset = .inputPriority

            do {
                let input = try AVCaptureDeviceInput(device: camera)
                if self.captureSession.canAddInput(input) { self.captureSession.addInput(input) }
            } catch {
                DispatchQueue.main.async { self.error = .setupFailed(error) }
                self.captureSession.commitConfiguration()
                return
            }

            self.videoOutput.setSampleBufferDelegate(self, queue: self.outputQueue)
            // 4:2:0 YUV is 1.5 bytes/pixel vs BGRA's 4 — the pre-trigger ring costs
            // ~2.7× less memory and the H.264 encoder ingests it without conversion.
            // CIImage (model preprocessing + previews) reads it natively.
            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            ]
            self.videoOutput.alwaysDiscardsLateVideoFrames = true

            if self.captureSession.canAddOutput(self.videoOutput) {
                self.captureSession.addOutput(self.videoOutput)
            }

            // Rotate output so pixel data is portrait-oriented
            if let conn = self.videoOutput.connection(with: .video) {
                if #available(iOS 17.0, *) {
                    if conn.isVideoRotationAngleSupported(90) {
                        conn.videoRotationAngle = 90
                    }
                } else {
                    if conn.isVideoOrientationSupported {
                        conn.videoOrientation = .portrait
                    }
                }
            }

            self.captureSession.commitConfiguration()

            // Set format & fps now that the session is configured
            self.applyFormatAndFrameRate(to: camera)

            self.captureSession.startRunning()
            DispatchQueue.main.async {
                self.isSessionRunning = self.captureSession.isRunning
            }
        }
    }

    // MARK: - Session Control

    func startSession() {
        sessionQueue.async { [weak self] in
            guard let self, !self.captureSession.isRunning else { return }
            self.captureSession.startRunning()
            DispatchQueue.main.async { self.isSessionRunning = self.captureSession.isRunning }
        }
    }

    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self, self.captureSession.isRunning else { return }
            self.captureSession.stopRunning()
            DispatchQueue.main.async { self.isSessionRunning = self.captureSession.isRunning }
        }
    }

    func setFrameHandler(_ handler: @escaping (CMSampleBuffer) -> Void) {
        frameHandler = handler
    }

    func updateConfiguration(_ newConfig: RecordingConfiguration) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.configuration = newConfig
            // Only update the device format & fps — no session preset changes
            // (session is in .inputPriority mode so it won't fight us)
            if let camera = (self.captureSession.inputs.first as? AVCaptureDeviceInput)?.device {
                self.applyFormatAndFrameRate(to: camera)
            }
        }
    }

    // MARK: - Format / Frame Rate

    private func applyFormatAndFrameRate(to camera: AVCaptureDevice) {
        let targetFPS = configuration.frameRate.value
        let targetDim = CMVideoDimensions(
            width:  Int32(configuration.resolution.dimensions.width),
            height: Int32(configuration.resolution.dimensions.height))

        // Find a format that matches the target resolution AND supports the target FPS.
        // Fall back progressively: exact match → any format supporting target FPS → best available.
        let candidates = camera.formats.filter { fmt in
            let dim = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
            return dim.width == targetDim.width && dim.height == targetDim.height
        }
        let best: AVCaptureDevice.Format?
        if let f = candidates.first(where: { $0.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= targetFPS } }) {
            best = f
        } else if let f = candidates.first {
            // Resolution matched but FPS won't be exact; still use it
            best = f
        } else {
            // Fallback: pick any format that supports the desired FPS
            best = camera.formats.first { $0.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= targetFPS } }
        }

        guard let format = best else {
            print("⚠️ [CriClips] No suitable camera format found for \(configuration.resolution.displayName) @\(targetFPS)fps")
            return
        }

        // Clamp FPS to a range the chosen format actually supports.
        // Find the highest max FPS that is ≤ targetFPS, then verify it fits in a supported range.
        let supportedRanges = format.videoSupportedFrameRateRanges
        let candidateFPS = supportedRanges.map(\.maxFrameRate).filter { $0 <= targetFPS }.max()
                        ?? supportedRanges.map(\.maxFrameRate).min()
                        ?? targetFPS
        // Confirm the candidate is inside a continuous supported range
        let actualFPS = supportedRanges.contains(where: {
            $0.minFrameRate <= candidateFPS && candidateFPS <= $0.maxFrameRate
        }) ? candidateFPS : (supportedRanges.map(\.maxFrameRate).min() ?? 30.0)

        do {
            try camera.lockForConfiguration()
            camera.activeFormat = format
            let timescale = CMTimeScale(Int(actualFPS.rounded()))
            let dur = CMTime(value: 1, timescale: timescale)
            camera.activeVideoMinFrameDuration = dur
            camera.activeVideoMaxFrameDuration = dur
            camera.unlockForConfiguration()
            print("✅ [CriClips] Camera: \(configuration.resolution.displayName) @\(Int(actualFPS))fps")
        } catch {
            print("⚠️ [CriClips] Could not lock camera for configuration: \(error)")
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        internalFrameCount += 1
        if internalFrameCount % 30 == 0 {
            let fc = internalFrameCount
            let dc = internalDroppedCount
            DispatchQueue.main.async {
                self.frameCount = fc
                self.droppedFrameCount = dc
            }
        }
        frameHandler?(sampleBuffer)
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didDrop sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        internalDroppedCount += 1
    }
}
