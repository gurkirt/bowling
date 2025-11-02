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
    private let visionQueue = DispatchQueue(label: "model.vision.queue")
    private let ciContext = CIContext()
    private var previewCounter = 0

    var onTriggerDetected: (() -> Void)?

    init() {
        loadModel()
    }

    private func loadModel() {
        print("🔵 loadModel() called")
        
        // Try to load compiled model from bundle (.mlmodelc)
        if let url = Bundle.main.url(forResource: "best_model", withExtension: "mlmodelc") {
            print("✅ Found .mlmodelc at: \(url)")
            do {
                model = try MLModel(contentsOf: url)
                // Don't use Vision - model expects tensor input, not image
                print("✅ Model loaded successfully (tensor input mode)")
                print("📋 Model description: \(model!.modelDescription)")
                DispatchQueue.main.async { self.isModelLoaded = true }
                return
            } catch {
                print("❌ Failed to load MLModel: \(error)")
            }
        }
        
        // Fallback: attempt to load .mlpackage
        if let pkg = Bundle.main.url(forResource: "best_model", withExtension: "mlpackage") {
            print("✅ Found .mlpackage at: \(pkg)")
            do {
                model = try MLModel(contentsOf: pkg)
                print("✅ Model loaded successfully (tensor input mode)")
                print("📋 Model description: \(model!.modelDescription)")
                DispatchQueue.main.async { self.isModelLoaded = true }
                return
            } catch {
                print("❌ Failed to load MLPackage: \(error)")
            }
        }
        
        print("❌ Model file not found in bundle")
    }

    func processFrame(_ sampleBuffer: CMSampleBuffer) {
        guard isModelLoaded, let model = model else { return }

        // Convert to CIImage
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        image = image.oriented(.right) // Rotate incoming buffer to match UI preview orientation
        let extent = image.extent
        let width = extent.width
        let height = extent.height

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

        // Keep a small preview (downsample for UI)
        previewCounter &+= 1
        if previewCounter % 4 == 0 {
            if let ui = renderUIImage(from: image, targetSize: CGSize(width: 160, height: 160)) {
                DispatchQueue.main.async { self.previewImage = ui }
            }
        }

        // 3) Resize to 256x256
    let scale = 256.0 / Double(squareSide)
    let transform = CGAffineTransform(scaleX: scale, y: scale)
    let resized = image.transformed(by: transform)
        
        // 4) Convert to pixel buffer for normalization
        guard let resizedPixelBuffer = createPixelBuffer(from: resized, width: 256, height: 256) else {
            print("❌ Failed to create pixel buffer")
            return
        }
        
        // 5) Normalize and run prediction on background queue
        visionQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                // Create MLMultiArray input (assuming [1, 3, 256, 256] CHW format)
                let inputArray = try self.normalizePixelBuffer(resizedPixelBuffer)
                
                // Get input feature name from model
                let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "input"
                let input = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
                
                // Run prediction
                let output = try model.prediction(from: input)
                
                // Parse output (assuming classification with class names)
                if let outputFeature = model.modelDescription.outputDescriptionsByName.keys.first,
                   let outputValue = output.featureValue(for: outputFeature) {
                    
                    if let multiArray = outputValue.multiArrayValue {
                        // Assuming 2 classes: [action_score, no_action_score]
                        let actionScore = Float(truncating: multiArray[0])
                        let noActionScore = Float(truncating: multiArray[1])
                        self.evaluateDecision(actionScore: actionScore, noActionScore: noActionScore)
                    }
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
    
    private func normalizePixelBuffer(_ pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
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
        
        // ImageNet mean and std
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        
        let data = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * 4
                
                // BGRA format
                let b = Float(data[pixelOffset]) / 255.0
                let g = Float(data[pixelOffset + 1]) / 255.0
                let r = Float(data[pixelOffset + 2]) / 255.0
                
                // Normalize and store in CHW format
                let idx = y * width + x
                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: (r - mean[0]) / std[0])
                array[[0, 1, y, x] as [NSNumber]] = NSNumber(value: (g - mean[1]) / std[1])
                array[[0, 2, y, x] as [NSNumber]] = NSNumber(value: (b - mean[2]) / std[2])
            }
        }
        
        return array
    }

    func resetTrigger() {
        recentActionWins.removeAll()
    }
}
