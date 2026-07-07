//
//  ModelProcessor.swift
//  CricReel
//
//  Runs the bundled CoreML cricket-action classifier on each camera frame.
//  Pre-processes: crop top 32%, centered square, resize 256×256, pack NCHW float32.
//  Sliding-window trigger: ≥4 of last 8 frames favour "action" → fires onTriggerDetected.
//
//  Trimmed from CriClips: debug preview images and on-device parity testing removed.
//

import CoreML
import AVFoundation
import CoreImage

class ModelProcessor: ObservableObject {
    @Published var isModelLoaded = false
    @Published var lastActionScore: Float = 0
    @Published var lastNoActionScore: Float = 0

    // ── Run / pause ────────────────────────────────────────────────────────
    @Published var isRunning: Bool = true {
        didSet {
            if !isRunning {
                DispatchQueue.main.async {
                    self.lastActionScore   = 0
                    self.lastNoActionScore = 0
                    self.recentActionWins.removeAll()
                }
            }
        }
    }
    /// Human-readable description of the MLComputeUnits the model was loaded with.
    @Published var computeUnitsLabel: String = ""

    // ── Configurable trigger settings ──────────────────────────────────────
    @Published var windowSize: Int = 8 {
        didSet { DispatchQueue.main.async { self.recentActionWins.removeAll() } }
    }
    @Published var requiredActionCount: Int = 4
    /// Minimum action score for a frame to count as an "action win" (0.0–1.0).
    @Published var scoreThreshold: Double = 0.5
    /// Run inference on every Nth camera frame (1 = every frame, 2 = every other, etc.).
    @Published var inferenceFrameInterval: Int = 1

    private var recentActionWins: [Bool] = []
    private var model: MLModel?
    private let inputFeatureName = "image"
    private var outputFeatureName = "var_734"

    private let visionQueue = DispatchQueue(label: "cricreel.model.queue", qos: .userInteractive)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var inferenceCounter = 0

    var onTriggerDetected: (() -> Void)?

    init() {
        loadModel()
    }

    // MARK: - Model Loading

    private func loadModel() {
        let loadQueue = DispatchQueue(label: "cricreel.model.load", qos: .userInitiated)
        loadQueue.async { [weak self] in
            guard let self else { return }
            for bundle in [Bundle.main, Bundle(for: ModelProcessor.self)] {
                if let url = bundle.url(forResource: "fastvit_sa12_exp21", withExtension: "mlpackage") {
                    self.tryLoad(url: url)
                    return
                }
                if let url = bundle.url(forResource: "fastvit_sa12_exp21", withExtension: "mlmodelc") {
                    self.tryLoad(url: url)
                    return
                }
            }
            print("❌ [CricReel] fastvit_sa12_exp21.mlpackage not found in bundle")
        }
    }

    private func tryLoad(url: URL) {
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .all  // ANE + GPU + CPU for best on-device performance
            let loaded = try MLModel(contentsOf: url, configuration: cfg)
            readOutputNameFromMetadata(loaded)
            let label = Self.computeUnitsString(cfg.computeUnits)
            DispatchQueue.main.async { [weak self] in
                self?.model = loaded
                self?.isModelLoaded = true
                self?.computeUnitsLabel = label
                print("✅ [CricReel] Model loaded: \(url.lastPathComponent) | compute: \(label)")
            }
        } catch {
            print("❌ [CricReel] Failed to load model at \(url.lastPathComponent): \(error)")
        }
    }

    private func readOutputNameFromMetadata(_ model: MLModel) {
        if let user = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
           let name = user["output_name"], !name.isEmpty {
            outputFeatureName = name
        }
    }

    private static func computeUnitsString(_ units: MLComputeUnits) -> String {
        switch units {
        case .all:                return "ANE + GPU + CPU"
        case .cpuAndNeuralEngine: return "ANE + CPU"
        case .cpuAndGPU:          return "GPU + CPU"
        case .cpuOnly:            return "CPU Only"
        @unknown default:         return "Auto"
        }
    }

    // MARK: - Frame Processing

    func processFrame(_ sampleBuffer: CMSampleBuffer) {
        guard isModelLoaded, isRunning, let model else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Throttle: skip frames according to inferenceFrameInterval
        inferenceCounter &+= 1
        guard inferenceCounter % max(1, inferenceFrameInterval) == 0 else { return }

        // Build CIImage from camera pixel buffer
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = image.extent
        let w = extent.width
        let h = extent.height

        // ── Step 1: Remove top 32%, keep bottom 68% ──────────────────────────
        // CIImage origin is bottom-left, so keeping y=minY … keptHeight gives the visual bottom.
        let keptH = h * 0.68
        image = image.cropped(to: CGRect(x: extent.minX, y: extent.minY, width: w, height: keptH))

        // ── Step 2: Centered square crop ──────────────────────────────────────
        let cropExt = image.extent
        let side = min(cropExt.width, cropExt.height)
        let squareRect = CGRect(
            x: cropExt.midX - side / 2,
            y: cropExt.midY - side / 2,
            width: side, height: side)
        image = image.cropped(to: squareRect)

        // ── Step 3: Resize to 256 × 256 ───────────────────────────────────────
        let scale = 256.0 / side
        let resized = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        // ── Step 4: Pack into NCHW float32 and run inference ──────────────────
        guard let pb = createPixelBuffer(from: resized, width: 256, height: 256) else { return }

        visionQueue.async { [weak self] in
            guard let self else { return }
            do {
                let inputArray = try self.makeNCHWArray(from: pb)
                let input = try MLDictionaryFeatureProvider(dictionary: [self.inputFeatureName: inputArray])
                let output = try model.prediction(from: input)

                let outKey = output.featureNames.contains(self.outputFeatureName)
                    ? self.outputFeatureName
                    : (model.modelDescription.outputDescriptionsByName.keys.first ?? self.outputFeatureName)

                if let fv = output.featureValue(for: outKey), let arr = fv.multiArrayValue {
                    let noAction = Float(truncating: arr[0])
                    let action   = Float(truncating: arr[1])
                    self.evaluateDecision(actionScore: action, noActionScore: noAction)
                }
            } catch {
                print("❌ [CricReel] Prediction error: \(error)")
            }
        }
    }

    // MARK: - Sliding Window Trigger

    private func evaluateDecision(actionScore: Float, noActionScore: Float) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.lastActionScore = actionScore
            self.lastNoActionScore = noActionScore
            let isWin = actionScore >= Float(self.scoreThreshold)
            self.recentActionWins.append(isWin)
            if self.recentActionWins.count > self.windowSize {
                self.recentActionWins.removeFirst()
            }
            let wins = self.recentActionWins.filter { $0 }.count
            let threshold = min(self.requiredActionCount, self.windowSize)
            if self.recentActionWins.count == self.windowSize && wins >= threshold {
                self.onTriggerDetected?()
                self.recentActionWins.removeAll()
            }
        }
    }

    func resetTrigger() {
        recentActionWins.removeAll()
    }

    // MARK: - Image Utilities

    private func createPixelBuffer(from ciImage: CIImage, width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        var pb: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { return nil }
        let translated = ciImage.transformed(
            by: CGAffineTransform(translationX: -ciImage.extent.origin.x,
                                  y: -ciImage.extent.origin.y))
        ciContext.render(translated, to: buffer)
        return buffer
    }

    /// Pack a 256×256 BGRA pixel buffer into an NCHW float32 MLMultiArray [1,3,256,256].
    /// Applies ImageNet normalisation: (pixel/255 − mean) / std  (matches the Python export pipeline).
    private func makeNCHWArray(from pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width       = CVPixelBufferGetWidth(pixelBuffer)
        let height      = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base  = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NSError(domain: "CricReel.ModelProcessor", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Nil pixel buffer base address"])
        }

        let array = try MLMultiArray(shape: [1, 3, height as NSNumber, width as NSNumber],
                                     dataType: .float32)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let stride = width * height

        // ImageNet normalisation constants (R, G, B order)
        let meanR: Float32 = 0.485, meanG: Float32 = 0.456, meanB: Float32 = 0.406
        let stdR:  Float32 = 0.229, stdG:  Float32 = 0.224, stdB:  Float32 = 0.225

        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(array.dataPointer))

        for y in 0 ..< height {
            for x in 0 ..< width {
                let off = y * bytesPerRow + x * 4
                ptr[0 * stride + y * width + x] = (Float32(pixels[off + 2]) / 255.0 - meanR) / stdR // R
                ptr[1 * stride + y * width + x] = (Float32(pixels[off + 1]) / 255.0 - meanG) / stdG // G
                ptr[2 * stride + y * width + x] = (Float32(pixels[off + 0]) / 255.0 - meanB) / stdB // B
            }
        }
        return array
    }
}
