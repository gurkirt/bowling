//
//  ModelProcessor.swift
//  CriClips
//
//  Runs the bundled CoreML cricket-action classifier on each camera frame.
//  Pre-processes: two crops beside the calibrated stumps (StumpCalibration.swift), the right
//  one mirrored, each resized to 160×256 and packed as one NCHW float32 batch of 2.
//  A frame's action score is the higher of the two crops'.
//  Sliding-window trigger: ≥4 of last 8 frames favour "action" → fires onTriggerDetected.
//

import CoreML
import AVFoundation
import CoreImage
import UIKit
import Accelerate

class ModelProcessor: ObservableObject {
    @Published var isModelLoaded = false
    /// Small thumbnail displayed in the live UI (updated every 4 frames).
    @Published var previewImage: UIImage?
    /// The two model-input crops side by side (left, mirrored right), shown in the main UI.
    @Published var bigPreviewImage: UIImage?
    /// Downscaled full camera frame, published while `publishesFullFrame` is set (calibration).
    @Published var fullFramePreview: UIImage?
    var publishesFullFrame = false
    @Published var lastActionScore: Float = 0
    @Published var lastNoActionScore: Float = 0

    // ── Run / pause ────────────────────────────────────────────────────────
    @Published var isRunning: Bool = true {
        didSet {
            if !isRunning {
                DispatchQueue.main.async {
                    self.lastActionScore    = 0
                    self.lastNoActionScore  = 0
                    self.previewImage       = nil
                    self.bigPreviewImage    = nil
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

    /// Where the stumps are; set by the calibration screen, read on the capture thread.
    var stumpBox: StumpBox {
        get { stumpLock.lock(); defer { stumpLock.unlock() }; return _stumpBox }
        set { stumpLock.lock(); _stumpBox = newValue; stumpLock.unlock() }
    }
    private var _stumpBox = StumpBox.load()
    private let stumpLock = NSLock()

    private var recentActionWins: [Bool] = []
    private var model: MLModel?
    private let inputFeatureName = "image"
    private var outputFeatureName = "var_734"

    private let visionQueue = DispatchQueue(label: "criclips.model.queue", qos: .userInteractive)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var frameCounter = 0
    private var inferenceCounter = 0
    private var fullFrameCounter = 0

    var onTriggerDetected: (() -> Void)?

    init() {
        loadModel()
    }

    // MARK: - Model Loading

    private func loadModel() {
        let loadQueue = DispatchQueue(label: "criclips.model.load", qos: .userInitiated)
        loadQueue.async { [weak self] in
            guard let self else { return }
            for bundle in [Bundle.main, Bundle(for: ModelProcessor.self)] {
                if let url = bundle.url(forResource: "fastvit_sa12_s04", withExtension: "mlpackage") {
                    self.tryLoad(url: url)
                    return
                }
                if let url = bundle.url(forResource: "fastvit_sa12_s04", withExtension: "mlmodelc") {
                    self.tryLoad(url: url)
                    return
                }
            }
            print("❌ [CriClips] fastvit_sa12_s04.mlpackage not found in bundle")
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
                print("✅ [CriClips] Model loaded: \(url.lastPathComponent) | compute: \(label)")
            }
        } catch {
            print("❌ [CriClips] Failed to load model at \(url.lastPathComponent): \(error)")
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
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        publishFullFrameIfNeeded(image)
        guard isModelLoaded, isRunning, let model else { return }

        // Throttle: skip frames according to inferenceFrameInterval
        inferenceCounter &+= 1
        guard inferenceCounter % max(1, inferenceFrameInterval) == 0 else { return }

        frameCounter &+= 1

        // ── Step 1: Two crops beside the stumps, 160×256 each ─────────────────
        let crops = stumpCrops(from: image)
        let w = StumpCrops.inputWidth, h = StumpCrops.inputHeight

        // ── Step 2: Update UI preview every 4 frames ─────────────────────────
        if frameCounter % 4 == 0 {
            let pair = crops.right.transformed(by: CGAffineTransform(translationX: CGFloat(w), y: 0))
                .composited(over: crops.left)
            if let ui = renderUIImage(from: pair, targetSize: CGSize(width: 2 * w, height: h)) {
                DispatchQueue.main.async { [weak self] in
                    self?.bigPreviewImage = ui
                    self?.previewImage = ui
                }
            }
        }

        // ── Step 3: Pack both crops into one NCHW batch and run inference ─────
        guard let left = createPixelBuffer(from: crops.left, width: w, height: h),
              let right = createPixelBuffer(from: crops.right, width: w, height: h) else { return }

        visionQueue.async { [weak self] in
            guard let self else { return }
            do {
                let action = try self.predictAction(model, left, right)
                self.evaluateDecision(actionScore: action, noActionScore: 1 - action)
            } catch {
                print("❌ [CriClips] Prediction error: \(error)")
            }
        }
    }

    /// Action score of a frame: the higher of its two crops' (probs is [2, 2], one row per crop).
    private func predictAction(_ model: MLModel, _ left: CVPixelBuffer, _ right: CVPixelBuffer) throws -> Float {
        let inputArray = try makeNCHWArray(from: [left, right])
        let input = try MLDictionaryFeatureProvider(dictionary: [inputFeatureName: inputArray])
        let output = try model.prediction(from: input)
        let outKey = output.featureNames.contains(outputFeatureName)
            ? outputFeatureName
            : (model.modelDescription.outputDescriptionsByName.keys.first ?? outputFeatureName)
        guard let arr = output.featureValue(for: outKey)?.multiArrayValue else {
            throw NSError(domain: "CriClips.ModelProcessor", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "No output \(outKey)"])
        }
        return max(arr[[0, 1] as [NSNumber]].floatValue, arr[[1, 1] as [NSNumber]].floatValue)
    }

    /// The two model inputs: crops either side of the stumps scaled to 160×256, the right
    /// one mirrored so the bowler is always left of the stumps. Parts of a crop outside the
    /// frame are black, as in training.
    private func stumpCrops(from image: CIImage) -> (left: CIImage, right: CIImage) {
        let size = image.extent.size
        let rects = StumpCrops.rects(for: stumpBox, in: size)
        let padded = image.composited(over: CIImage(color: .black))
        let w = CGFloat(StumpCrops.inputWidth), h = CGFloat(StumpCrops.inputHeight)
        func crop(_ r: CGRect, mirror: Bool) -> CIImage {
            // Top-left pixel rect -> CIImage's bottom-left origin.
            let ci = CGRect(x: r.minX, y: size.height - r.maxY, width: r.width, height: r.height).integral
            var out = padded.cropped(to: ci)
                .transformed(by: CGAffineTransform(translationX: -ci.minX, y: -ci.minY))
                .applyingFilter("CILanczosScaleTransform", parameters: [
                    kCIInputScaleKey: h / ci.height,
                    kCIInputAspectRatioKey: (w / ci.width) / (h / ci.height)])
                .cropped(to: CGRect(x: 0, y: 0, width: w, height: h))
            if mirror { out = out.transformed(by: CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: w, ty: 0)) }
            return out
        }
        return (crop(rects.left, mirror: false), crop(rects.right, mirror: true))
    }

    /// Full frame for the calibration screen, every 4th camera frame.
    private func publishFullFrameIfNeeded(_ image: CIImage) {
        guard publishesFullFrame else { return }
        fullFrameCounter &+= 1
        guard fullFrameCounter % 4 == 0 else { return }
        let size = CGSize(width: 360, height: (360 * image.extent.height / image.extent.width).rounded())
        if let ui = renderUIImage(from: image, targetSize: size) {
            DispatchQueue.main.async { [weak self] in self?.fullFramePreview = ui }
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

    // MARK: - Static Image Scoring (for on-device parity testing)

    /// Score a UIImage through the same CoreML pipeline as live frames.
    /// - Parameters:
    ///   - uiImage: The image to score.
    ///   - applyPreprocessing: `true` → a raw camera frame (e.g. debug_frame_full.png 1080×1920):
    ///     cut the two stump crops from the current calibration. `false` → an already-cropped
    ///     model input, resized to 160×256 and scored as both crops.
    ///   - completion: Called on the main queue with (noActionScore, actionScore).
    func scoreImage(_ uiImage: UIImage,
                    applyPreprocessing: Bool,
                    completion: @escaping (Float, Float) -> Void) {
        guard isModelLoaded, let model else { return }
        guard let cgImage = uiImage.cgImage else { return }

        let image = CIImage(cgImage: cgImage)
        let w = StumpCrops.inputWidth, h = StumpCrops.inputHeight
        let crops: (left: CIImage, right: CIImage)
        if applyPreprocessing {
            crops = stumpCrops(from: image)
        } else {
            let resized = image.transformed(by: CGAffineTransform(
                scaleX: CGFloat(w) / image.extent.width, y: CGFloat(h) / image.extent.height))
            crops = (resized, resized)
        }
        guard let left = createPixelBuffer(from: crops.left, width: w, height: h),
              let right = createPixelBuffer(from: crops.right, width: w, height: h) else { return }

        visionQueue.async { [weak self] in
            guard let self else { return }
            do {
                let action = try self.predictAction(model, left, right)
                DispatchQueue.main.async { completion(1 - action, action) }
            } catch {
                print("❌ [CriClips] scoreImage error: \(error)")
            }
        }
    }

    // MARK: - Image Utilities

    private func renderUIImage(from ciImage: CIImage, targetSize: CGSize) -> UIImage? {
        let translated = ciImage.transformed(
            by: CGAffineTransform(translationX: -ciImage.extent.origin.x,
                                  y: -ciImage.extent.origin.y))
        let sx = targetSize.width  / translated.extent.width
        let sy = targetSize.height / translated.extent.height
        let scaled = translated.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        guard let cg = ciContext.createCGImage(scaled, from: CGRect(origin: .zero, size: targetSize))
        else { return nil }
        return UIImage(cgImage: cg)
    }

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

    /// Pack same-sized BGRA buffers (one per crop) into an NCHW float32 batch [n, 3, H, W].
    private func makeNCHWArray(from pixelBuffers: [CVPixelBuffer]) throws -> MLMultiArray {
        let width  = CVPixelBufferGetWidth(pixelBuffers[0])
        let height = CVPixelBufferGetHeight(pixelBuffers[0])
        let array = try MLMultiArray(shape: [pixelBuffers.count as NSNumber, 3, height as NSNumber, width as NSNumber],
                                     dataType: .float32)
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(array.dataPointer))
        let stride = width * height

        // ImageNet normalisation, vectorised: out = px * 1/(255·std) + (−mean/std).
        let means: [Float] = [0.485, 0.456, 0.406]          // R, G, B
        let stds:  [Float] = [0.229, 0.224, 0.225]
        let bgraOffset = [2, 1, 0]                          // R, G, B positions in BGRA

        for (n, pixelBuffer) in pixelBuffers.enumerated() {
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                throw NSError(domain: "CriClips.ModelProcessor", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Nil pixel buffer base address"])
            }
            let pixels = base.assumingMemoryBound(to: UInt8.self)
            for c in 0 ..< 3 {
                var scale = Float(1.0) / (255.0 * stds[c])
                var bias  = -means[c] / stds[c]
                for y in 0 ..< height {
                    let src = pixels + y * bytesPerRow + bgraOffset[c]
                    let dst = ptr + (n * 3 + c) * stride + y * width
                    vDSP_vfltu8(src, 4, dst, 1, vDSP_Length(width))
                    vDSP_vsmsa(dst, 1, &scale, &bias, dst, 1, vDSP_Length(width))
                }
            }
        }
        return array
    }
}
