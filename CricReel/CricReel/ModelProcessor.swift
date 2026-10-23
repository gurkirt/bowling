//
//  ModelProcessor.swift
//  CricReel
//
//  Runs the bundled CoreML cricket-action classifier on each camera frame.
//  Pre-processes: two crops beside the calibrated stumps (StumpCalibration.swift), the right
//  one mirrored, each resized to 160×256 and packed as one NCHW float32 batch of 2.
//  A frame's action score is the higher of the two crops'.
//  Publishes the two crops side by side as the preview (exactly what the model sees).
//  Sliding-window trigger: ≥N of last W frames favour "action" → fires onTriggerDetected.
//

import CoreML
import AVFoundation
import CoreImage
import UIKit
import Accelerate

class ModelProcessor: ObservableObject {
    @Published var isModelLoaded = false
    /// Downscaled full camera frame for the live UI.
    @Published var previewImage: UIImage?
    /// Downscaled full camera frame, published while `publishesFullFrame` is set (calibration).
    @Published var fullFramePreview: UIImage?
    var publishesFullFrame = false
    @Published var lastActionScore: Float = 0
    @Published var lastNoActionScore: Float = 0

    @Published var isRunning: Bool = true {
        didSet {
            if !isRunning {
                DispatchQueue.main.async {
                    self.lastActionScore   = 0
                    self.lastNoActionScore = 0
                    self.previewImage      = nil
                    self.recentActionWins.removeAll()
                }
            }
        }
    }
    @Published var computeUnitsLabel: String = ""

    // Configurable trigger settings
    @Published var windowSize: Int = 8 {
        didSet { DispatchQueue.main.async { self.recentActionWins.removeAll() } }
    }
    @Published var requiredActionCount: Int = 4
    @Published var scoreThreshold: Double = 0.5
    @Published var inferenceFrameInterval: Int = 1
    /// Where the stumps are; set by the calibration screen, read on the capture thread.
    var stumpBox: StumpBox {
        get { stumpLock.lock(); defer { stumpLock.unlock() }; return _stumpBox }
        set { stumpLock.lock(); _stumpBox = newValue; stumpLock.unlock() }
    }
    private var _stumpBox = StumpBox.load()
    private let stumpLock = NSLock()

    /// Inference ceiling. Detection quality doesn't improve past ~30 scored frames/sec,
    /// so at 60 fps capture every other frame is skipped to save ANE/GPU power.
    private let maxInferenceFPS: Double = 30
    private var lastInferenceTime: CFTimeInterval = 0

    private var recentActionWins: [Bool] = []
    private var model: MLModel?
    private let inputFeatureName = "image"
    private var outputFeatureName = "var_734"

    private let visionQueue = DispatchQueue(label: "cricreel.model.queue", qos: .userInteractive)
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
        let loadQueue = DispatchQueue(label: "cricreel.model.load", qos: .userInitiated)
        loadQueue.async { [weak self] in
            guard let self else { return }
            for bundle in [Bundle.main, Bundle(for: ModelProcessor.self)] {
                if let url = bundle.url(forResource: "fastvit_sa12_s04", withExtension: "mlpackage") {
                    self.tryLoad(url: url); return
                }
                if let url = bundle.url(forResource: "fastvit_sa12_s04", withExtension: "mlmodelc") {
                    self.tryLoad(url: url); return
                }
            }
            print("❌ [CricReel] fastvit_sa12_s04.mlpackage not found in bundle")
        }
    }

    private func tryLoad(url: URL) {
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .all
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
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        publishFullFrameIfNeeded(image)
        guard isModelLoaded, isRunning, let model else { return }

        inferenceCounter &+= 1
        guard inferenceCounter % max(1, inferenceFrameInterval) == 0 else { return }

        // Cap inference at ~30 fps regardless of capture rate. The 0.9 factor absorbs
        // frame-timing jitter so 30 fps capture is never accidentally halved.
        let now = CACurrentMediaTime()
        guard now - lastInferenceTime >= (1.0 / maxInferenceFPS) * 0.9 else { return }
        lastInferenceTime = now

        frameCounter &+= 1

        let crops = stumpCrops(from: image)
        let w = StumpCrops.inputWidth, h = StumpCrops.inputHeight

        // Live preview (every 4 scored frames).
        if frameCounter % 4 == 0 {
            // Full frame (the stump guide is drawn over it by the view).
            let size = CGSize(width: 360, height: (360 * image.extent.height / image.extent.width).rounded())
            if let ui = renderUIImage(from: image, targetSize: size) {
                DispatchQueue.main.async { [weak self] in self?.previewImage = ui }
            }
        }

        guard let left = createPixelBuffer(from: crops.left, width: w, height: h),
              let right = createPixelBuffer(from: crops.right, width: w, height: h) else { return }

        visionQueue.async { [weak self] in
            guard let self else { return }
            do {
                let inputArray = try self.makeNCHWArray(from: [left, right])
                let input = try MLDictionaryFeatureProvider(dictionary: [self.inputFeatureName: inputArray])
                let output = try model.prediction(from: input)
                let outKey = output.featureNames.contains(self.outputFeatureName)
                    ? self.outputFeatureName
                    : (model.modelDescription.outputDescriptionsByName.keys.first ?? self.outputFeatureName)
                if let fv = output.featureValue(for: outKey), let arr = fv.multiArrayValue {
                    // probs [2, 2], one row per crop; the frame fires if either crop does.
                    let action = max(arr[[0, 1] as [NSNumber]].floatValue, arr[[1, 1] as [NSNumber]].floatValue)
                    self.evaluateDecision(actionScore: action, noActionScore: 1 - action)
                }
            } catch {
                print("❌ [CricReel] Prediction error: \(error)")
            }
        }
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

    func resetTrigger() { recentActionWins.removeAll() }

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
                throw NSError(domain: "CricReel.ModelProcessor", code: 1,
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
