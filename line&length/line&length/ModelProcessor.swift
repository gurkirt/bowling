//
//  ModelProcessor.swift
//  line&length
//
//  Created by Jean Daniel Browne on 22.10.2025.
//

import CoreML
import Vision
import AVFoundation
import CoreImage
import UIKit

class ModelProcessor: ObservableObject {
    @Published var isModelLoaded = false
    @Published var previewImage: UIImage?
    @Published var lastActionScore: Float = 0
    @Published var lastNoActionScore: Float = 0

    // Sliding window: need at least 5 of last 8 frames with actionScore > noActionScore
    private var recentActionWins: [Bool] = []
    private let windowSize = 8
    private let requiredActionCount = 4

    private var model: MLModel?
    // CoreML I/O feature names (defaults; can be overridden by model metadata)
    private let inputFeatureName = "image"
    private var outputFeatureName = "var_734" // will try to read from creatorDefined metadata: output_name
    private let visionQueue = DispatchQueue(label: "model.vision.queue")
    private let ciContext = CIContext()
    private var previewCounter = 0
    #if DEBUG
    private let debugSaveEnabled = true
    private let debugSaveLimit = 200
    private var debugSaved = 0
    private lazy var debugSaveDir: URL? = {
        let fm = FileManager.default
        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            let dir = docs.appendingPathComponent("DebugModelInputs", isDirectory: true)
            do { try fm.createDirectory(at: dir, withIntermediateDirectories: true) } catch {
                print("⚠️ Could not create debug dir: \(error)")
            }
            return dir
        }
        return nil
    }()
    #endif

    var onTriggerDetected: (() -> Void)?
    private var isRunningTests: Bool {
        return ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    init() {
        loadModel()
    }

    private func loadModel() {
        print("🔵 loadModel() called")
        
        // Try both the main bundle and test bundle for resources
        let bundles = [Bundle.main, Bundle(for: ModelProcessor.self)]
        for bundle in bundles {
            // Try compiled model first (.mlmodelc)
            if let url = bundle.url(forResource: "best_model", withExtension: "mlmodelc") {
                print("✅ Found .mlmodelc at: \(url)")
                do {
                    model = try MLModel(contentsOf: url)
                    print("✅ Model loaded successfully (tensor input mode)")
                    self.applyIOOverridesFromMetadata(model!)
                    print("📋 Model description: \(model!.modelDescription)")
                    DispatchQueue.main.async { self.isModelLoaded = true }
                    return
                } catch {
                    print("❌ Failed to load MLModel: \(error)")
                }
            }
            // Fallback: attempt to load package (.mlpackage)
            if let pkg = bundle.url(forResource: "best_model", withExtension: "mlpackage") {
                print("✅ Found .mlpackage at: \(pkg)")
                do {
                    model = try MLModel(contentsOf: pkg)
                    print("✅ Model loaded successfully (tensor input mode)")
                    self.applyIOOverridesFromMetadata(model!)
                    print("📋 Model description: \(model!.modelDescription)")
                    DispatchQueue.main.async { self.isModelLoaded = true }
                    return
                } catch {
                    print("❌ Failed to load MLPackage: \(error)")
                }
            }
        }
        
        print("❌ Model file not found in bundle")
    }

    private func applyIOOverridesFromMetadata(_ model: MLModel) {
        // Read creator-defined metadata written during export (e.g., output_name)
        let metadata = model.modelDescription.metadata
        if let user = metadata[.creatorDefinedKey] as? [String: String] {
            if let out = user["output_name"], !out.isEmpty {
                print("🔧 Using output feature from metadata: \(out)")
                self.outputFeatureName = out
            } else {
                print("ℹ️ No output_name in metadata; defaulting to \(self.outputFeatureName)")
            }
        } else {
            print("ℹ️ No creatorDefined metadata found; using default output feature \(self.outputFeatureName)")
        }
    }

    func processFrame(_ sampleBuffer: CMSampleBuffer) {
        guard isModelLoaded, let model = model else { return }

        // Convert to CIImage
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        // Rotate only if portrait to keep downstream crop consistent with Python path
        if image.extent.width < image.extent.height {
            image = image.oriented(.right)
        }
        var extent = image.extent
        var width = extent.width
        var height = extent.height

        if previewCounter == 0 {
            print("📐 Camera buffer oriented extent: \(extent)")
        }

        // Remove top 32% -> keep bottom 68% (origin is bottom-left)
        let topCropHeight = height * 0.32
        let keptHeight = height - topCropHeight
        let cropRect = CGRect(x: extent.minX,
                              y: extent.minY,
                              width: width,
                              height: keptHeight)
        image = image.cropped(to: cropRect)

        // Centered square crop from remaining area
        let postCropExtent = image.extent
        let squareSide = min(postCropExtent.width, postCropExtent.height)
        let squareRect = CGRect(x: postCropExtent.midX - squareSide / 2,
                                y: postCropExtent.midY - squareSide / 2,
                                width: squareSide,
                                height: squareSide)
        image = image.cropped(to: squareRect)

        // Keep a small preview (downsample for UI) — in tests, update every frame
        previewCounter &+= 1
        let shouldUpdatePreview = isRunningTests || (previewCounter % 4 == 0)
        if shouldUpdatePreview {
            if let ui = renderUIImage(from: image, targetSize: CGSize(width: 160, height: 160)) {
                DispatchQueue.main.async { self.previewImage = ui }
            }
        }

        // 3) Resize to 256x256
            let scale = 256.0 / Double(squareSide)
            let transform = CGAffineTransform(scaleX: scale, y: scale)
            let resized = image.transformed(by: transform)

            // Optionally save the exact 256x256 RGB input fed to the model for debugging
            #if DEBUG
            if debugSaveEnabled && debugSaved < debugSaveLimit, let dir = debugSaveDir {
                saveDebugInputImage(resized, to: dir, index: debugSaved)
                debugSaved += 1
            }
            #endif
        
        // 4) Convert to pixel buffer for tensor packing (no normalization here; model normalizes internally)
        guard let resizedPixelBuffer = createPixelBuffer(from: resized, width: 256, height: 256) else {
            print("❌ Failed to create pixel buffer")
            return
        }
        
        // 5) Pack pixels into NCHW float32 [0,255] and run prediction on background queue
        visionQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                // Create MLMultiArray input (CHW, raw pixels in [0,255])
                let inputArray = try self.makeInputArrayFromPixelBuffer(resizedPixelBuffer)
                let input = try MLDictionaryFeatureProvider(dictionary: [self.inputFeatureName: inputArray])
                
                // Run prediction
                let output = try model.prediction(from: input)
                
                // Parse output probabilities
                let outKey = output.featureNames.contains(self.outputFeatureName)
                    ? self.outputFeatureName
                    : (model.modelDescription.outputDescriptionsByName.keys.first ?? self.outputFeatureName)
                if let outputValue = output.featureValue(for: outKey), let multiArray = outputValue.multiArrayValue {
                    // Expected order: [no_action, action]
                    let noActionScore = Float(truncating: multiArray[0])
                    let actionScore = Float(truncating: multiArray[1])
                    self.evaluateDecision(actionScore: actionScore, noActionScore: noActionScore)
                }
            } catch {
                print("❌ Prediction error: \(error)")
            }
        }
    }

    private func evaluateDecision(actionScore: Float, noActionScore: Float) {
        DispatchQueue.main.async {
            self.lastActionScore = actionScore
            self.lastNoActionScore = noActionScore
            let isActionWin = actionScore > noActionScore
            self.recentActionWins.append(isActionWin)
            if self.recentActionWins.count > self.windowSize {
                self.recentActionWins.removeFirst()
            }
            let count = self.recentActionWins.filter { $0 }.count
            if self.recentActionWins.count == self.windowSize && count >= self.requiredActionCount {
                self.onTriggerDetected?()
                self.recentActionWins.removeAll()
            }
        }
    }

    private func renderUIImage(from ciImage: CIImage, targetSize: CGSize) -> UIImage? {
        // Translate the image so its extent starts at origin (0,0)
        let translatedImage = ciImage.transformed(by: CGAffineTransform(translationX: -ciImage.extent.origin.x, y: -ciImage.extent.origin.y))
        
        let scaleX = targetSize.width / translatedImage.extent.width
        let scaleY = targetSize.height / translatedImage.extent.height
        let scaled = translatedImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        guard let cg = ciContext.createCGImage(scaled, from: CGRect(origin: .zero, size: targetSize)) else { return nil }
        return UIImage(cgImage: cg)
    }

    #if DEBUG
    // Save the 256x256 CIImage as PNG without additional scaling
    private func saveDebugInputImage(_ ciImage: CIImage, to directory: URL, index: Int) {
        // Translate image so origin is at (0,0) before creating CGImage
        let translated = ciImage.transformed(by: CGAffineTransform(translationX: -ciImage.extent.origin.x,
                                                                   y: -ciImage.extent.origin.y))
        let rect = CGRect(origin: .zero, size: CGSize(width: 256, height: 256))
        guard let cg = ciContext.createCGImage(translated, from: rect) else { return }
        let ui = UIImage(cgImage: cg)
        guard let data = ui.pngData() else { return }
        let filename = String(format: "debug_input_%06d.png", index)
        let url = directory.appendingPathComponent(filename)
        do {
            try data.write(to: url)
            if index == 0 {
                print("🖼️ Saved debug input images to: \(directory.path)")
            }
        } catch {
            print("⚠️ Failed to save debug image: \(error)")
        }
    }
    #endif
    
    private func createPixelBuffer(from ciImage: CIImage, width: Int, height: Int) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        // Translate image to origin before rendering
        let translatedImage = ciImage.transformed(by: CGAffineTransform(translationX: -ciImage.extent.origin.x, y: -ciImage.extent.origin.y))
        ciContext.render(translatedImage, to: buffer)
        return buffer
    }
    
    // Pack BGRA pixel buffer into NCHW float32 array with raw pixel values in [0,255]
    // CoreML model performs ImageNet normalization internally.
    private func makeInputArrayFromPixelBuffer(_ pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NSError(domain: "ModelProcessor", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to get pixel buffer base address"])
        }
        
        // Create MLMultiArray [1, 3, 256, 256] for CHW format
        let shape = [1, 3, 256, 256] as [NSNumber]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        
        let data = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * 4
                
                // BGRA format
                let b = Float(data[pixelOffset])
                let g = Float(data[pixelOffset + 1])
                let r = Float(data[pixelOffset + 2])
                
                // Store in CHW format (no normalization)
                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: r)
                array[[0, 1, y, x] as [NSNumber]] = NSNumber(value: g)
                array[[0, 2, y, x] as [NSNumber]] = NSNumber(value: b)
            }
        }
        
        return array
    }

    func resetTrigger() {
        recentActionWins.removeAll()
    }
}
