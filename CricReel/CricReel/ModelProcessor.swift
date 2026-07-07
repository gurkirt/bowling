//
//  ModelProcessor.swift
//  CricReel
//
//  Runs the bundled CoreML cricket-action classifier on each camera frame.
//  Pre-processes: crop top 32%, centered square, resize 256×256, pack NCHW float32.
//  Publishes a higher-resolution square preview (the exact region fed to the model).
//  Sliding-window trigger: ≥N of last W frames favour "action" → fires onTriggerDetected.
//

import CoreML
import AVFoundation
import CoreImage
import UIKit

class ModelProcessor: ObservableObject {
    @Published var isModelLoaded = false
    /// Higher-resolution square crop (the model-input region) for the live UI.
    @Published var previewImage: UIImage?
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
    /// Side length (px) of the published preview crop.
    var previewResolution: CGFloat = 384

    private var recentActionWins: [Bool] = []
    private var model: MLModel?
    private let inputFeatureName = "image"
    private var outputFeatureName = "var_734"

    private let visionQueue = DispatchQueue(label: "cricreel.model.queue", qos: .userInteractive)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var frameCounter = 0
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
                    self.tryLoad(url: url); return
                }
                if let url = bundle.url(forResource: "fastvit_sa12_exp21", withExtension: "mlmodelc") {
                    self.tryLoad(url: url); return
                }
            }
            print("❌ [CricReel] fastvit_sa12_exp21.mlpackage not found in bundle")
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
        guard isModelLoaded, isRunning, let model else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        inferenceCounter &+= 1
        guard inferenceCounter % max(1, inferenceFrameInterval) == 0 else { return }
        frameCounter &+= 1

        var image = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = image.extent
        let w = extent.width
        let h = extent.height

        // Remove top 32%, keep bottom 68%.
        let keptH = h * 0.68
        image = image.cropped(to: CGRect(x: extent.minX, y: extent.minY, width: w, height: keptH))

        // Centered square crop.
        let cropExt = image.extent
        let side = min(cropExt.width, cropExt.height)
        let squareRect = CGRect(x: cropExt.midX - side / 2, y: cropExt.midY - side / 2,
                                width: side, height: side)
        let square = image.cropped(to: squareRect)

        // Higher-resolution preview of the model-input region (every 4 frames).
        if frameCounter % 4 == 0 {
            let target = CGSize(width: previewResolution, height: previewResolution)
            if let ui = renderUIImage(from: square, targetSize: target) {
                DispatchQueue.main.async { [weak self] in self?.previewImage = ui }
            }
        }

        // Resize to 256×256 for inference.
        let scale = 256.0 / side
        let resized = square.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
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

        let meanR: Float32 = 0.485, meanG: Float32 = 0.456, meanB: Float32 = 0.406
        let stdR:  Float32 = 0.229, stdG:  Float32 = 0.224, stdB:  Float32 = 0.225

        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(array.dataPointer))
        for y in 0 ..< height {
            for x in 0 ..< width {
                let off = y * bytesPerRow + x * 4
                ptr[0 * stride + y * width + x] = (Float32(pixels[off + 2]) / 255.0 - meanR) / stdR
                ptr[1 * stride + y * width + x] = (Float32(pixels[off + 1]) / 255.0 - meanG) / stdG
                ptr[2 * stride + y * width + x] = (Float32(pixels[off + 0]) / 255.0 - meanB) / stdB
            }
        }
        return array
    }
}
