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
    @Published var bigPreviewImage: UIImage?
    @Published var lastActionScore: Float = 0
    @Published var lastNoActionScore: Float = 0

    // Sliding window: need at least 5 of last 8 frames with actionScore > noActionScore
    private var recentActionWins: [Bool] = []
    private let windowSize = 8
    private let requiredActionCount = 4

    private var model: MLModel?
    private var modelURL: URL?
    // CoreML I/O feature names (defaults; can be overridden by model metadata)
    private let inputFeatureName = "image"
    private var outputFeatureName = "var_734" // will try to read from creatorDefined metadata: output_name
    private let visionQueue = DispatchQueue(label: "model.vision.queue")
    private let ciContext = CIContext()
    private var previewCounter = 0
    private var didRunDefaultRawTensors = false
    // Suppress main processing (camera and UIImage paths) and run raw tensor parity only
    private var suppressMainSession: Bool = true
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

    init() {
        loadModel()
    }

    private func loadModel() {
        #if DEBUG
        print("🔵 loadModel() called")
        #endif
        let loadQueue = DispatchQueue(label: "model.load.queue", qos: .userInitiated)
        loadQueue.async {
            // Try both the main bundle and test bundle for resources
            let bundles = [Bundle.main, Bundle(for: ModelProcessor.self)]
            for bundle in bundles {
                // Prefer .mlpackage to avoid stale compiled .mlmodelc
                if let pkg = bundle.url(forResource: "best_model", withExtension: "mlpackage") {
                    print("✅ Found .mlpackage at: \(pkg)")
                    do {
                        let cfg = MLModelConfiguration()
                        cfg.computeUnits = .cpuOnly // improve parity with Python reference
                        let loaded = try MLModel(contentsOf: pkg, configuration: cfg)
                        print("✅ Model loaded (mlpackage) with computeUnits=.cpuOnly")
                        self.applyIOOverridesFromMetadata(loaded)
                        DispatchQueue.main.async {
                            self.model = loaded
                            self.modelURL = pkg
                            self.isModelLoaded = true
                            self.runDefaultRawTensorsIfPresent()
                        }
                        return
                    } catch {
                        print("❌ Failed to load MLPackage: \(error)")
                    }
                }
                // Fallback: try compiled model (.mlmodelc)
                if let url = bundle.url(forResource: "best_model", withExtension: "mlmodelc") {
                    print("✅ Found .mlmodelc at: \(url)")
                    do {
                        let cfg = MLModelConfiguration()
                        cfg.computeUnits = .cpuOnly
                        let loaded = try MLModel(contentsOf: url, configuration: cfg)
                        print("✅ Model loaded (mlmodelc) with computeUnits=.cpuOnly")
                        self.applyIOOverridesFromMetadata(loaded)
                        DispatchQueue.main.async {
                            self.model = loaded
                            self.modelURL = url
                            self.isModelLoaded = true
                            self.runDefaultRawTensorsIfPresent()
                        }
                        return
                    } catch {
                        print("❌ Failed to load MLModel: \(error)")
                    }
                }
            }
            print("❌ Model file not found in bundle")
        }
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
        // Skip camera processing when focusing on raw tensor debugging
        if suppressMainSession { return }

        // Convert to CIImage
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        var extent = image.extent
        var width = extent.width
        var height = extent.height

        if previewCounter == 0 {
            print("📐 Camera buffer oriented extent: \(extent)")
        }

    // Remove TOP 32% -> keep BOTTOM 68% (match Python modellib.crop semantics).
    // CIImage coordinates have origin at bottom-left, so keeping bottom region means y = minY.
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
        let shouldUpdatePreview = (previewCounter % 4 == 0)
        if shouldUpdatePreview {
            // Force preview upright in portrait: if buffer is landscape, rotate right for display only
            let isPortraitBuffer = image.extent.height >= image.extent.width
            let previewCI: CIImage = isPortraitBuffer ? image : image.oriented(.right)
            if let ui = renderUIImage(from: previewCI, targetSize: CGSize(width: 160, height: 160)) {
                DispatchQueue.main.async { self.previewImage = ui }
            }
        }

        // 3) Resize to 256x256
            let scale = 256.0 / Double(squareSide)
            let transform = CGAffineTransform(scaleX: scale, y: scale)
            let resized = image.transformed(by: transform)

            // Update larger preview with the exact 256x256 model input (cropped + resized)
            if let big = renderUIImage(from: resized, targetSize: CGSize(width: 300, height: 300)) {
                DispatchQueue.main.async { self.bigPreviewImage = big }
            }

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
                // Debug: compute per-channel (R,G,B) mean/std (pre-norm, 0-255) from the actual model input tensor
                #if DEBUG
                if let channelStats = self.computeAllChannelStats(from: inputArray) {
                    let fmt: (Float) -> String = { String(format: "%.4f", $0) }
                    // Expect ordering R,G,B based on packing logic
                    if channelStats.count >= 3 {
                        let r = channelStats[0]; let g = channelStats[1]; let b = channelStats[2]
                        print("Channel stats (pre-norm 0-255): R(mean=\(fmt(r.mean)), std=\(fmt(r.std))) G(mean=\(fmt(g.mean)), std=\(fmt(g.std))) B(mean=\(fmt(b.mean)), std=\(fmt(b.std)))")
                    } else {
                        print("Channel stats: \(channelStats.map{ "(mean=\(fmt($0.mean)), std=\(fmt($0.std)))" }.joined(separator: ", ")))" )
                    }
                }
                #endif
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

    // MARK: - Debug: Load raw NCHW float32 tensor from Documents and run inference
    // Expects a .bin file containing 1*3*256*256 float32 values in NCHW order (little-endian),
    // optionally accompanied by a .json meta file with shape/dtype info (not required).
    // Auto-run dumped tensors if present (parity check without extra UI)
    private func runDefaultRawTensorsIfPresent() {
        guard !didRunDefaultRawTensors else { return }
        didRunDefaultRawTensors = true
        print("🧪 Debug mode: suppressing main session; running raw tensor parity checks if available…")
        let basenames = ["tensor_frame_000035", "tensor_frame_000045"]
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let dir = docs.appendingPathComponent("DebugModelInputs", isDirectory: true)
        for name in basenames {
            let binURL = dir.appendingPathComponent(name).appendingPathExtension("bin")
            if fm.fileExists(atPath: binURL.path) {
                print("▶️ Auto-loading raw tensor (Documents): \(name)")
                self.processRawTensorFromDocuments(basename: name)
            } else if hasAsset(named: name) || hasBundleResource(named: name, ext: "bin") {
                print("▶️ Auto-loading raw tensor (Assets/Bundle): \(name)")
                self.processRawTensorFromAssets(basename: name)
            } else {
                print("ℹ️ Raw tensor not found in Documents or Assets: \(name)")
            }
        }
    }

    // Try to load raw tensor from Assets (NSDataAsset) or bundled resource files
    private func processRawTensorFromAssets(basename: String) {
        // Load .bin
        guard let binData = loadAssetData(named: basename) ?? loadBundleResource(named: basename, ext: "bin") else {
            print("❌ Asset/bin not found for \(basename)")
            return
        }
        // Load optional .json for shape override
        let jsonData = loadAssetData(named: basename + "_json")
            ?? loadAssetData(named: basename + "_meta")
            ?? loadBundleResource(named: basename, ext: "json")

        var shape: [Int] = [1, 3, 256, 256]
        if let jd = jsonData, let s = readShape(from: jd) { shape = s }

        do {
            guard let array = try makeInputArrayFromRawFloat32(data: binData, shape: shape) else {
                print("❌ Failed to create MLMultiArray from asset tensor")
                return
            }
            if let channelStats = computeAllChannelStats(from: array) {
                let fmt: (Float) -> String = { String(format: "%.4f", $0) }
                if channelStats.count >= 3 {
                    let r = channelStats[0]; let g = channelStats[1]; let b = channelStats[2]
                    print("ASSET Tensor Channel stats (pre-norm 0-255): R(mean=\(fmt(r.mean)), std=\(fmt(r.std))) G(mean=\(fmt(g.mean)), std=\(fmt(g.std))) B(mean=\(fmt(b.mean)), std=\(fmt(b.std)))")
                }
            }
            try predict(with: array, sourceTag: "ASSET Tensor")
        } catch {
            print("❌ Prediction error (asset tensor): \(error)")
        }
    }

    private func hasAsset(named: String) -> Bool {
        return NSDataAsset(name: named) != nil
    }
    private func hasBundleResource(named: String, ext: String) -> Bool {
        return Bundle.main.url(forResource: named, withExtension: ext) != nil
    }
    private func loadAssetData(named: String) -> Data? {
        if let asset = NSDataAsset(name: named) { return asset.data }
        return nil
    }
    private func loadBundleResource(named: String, ext: String) -> Data? {
        guard let url = Bundle.main.url(forResource: named, withExtension: ext) else { return nil }
        return try? Data(contentsOf: url)
    }
    private func readShape(from jsonData: Data) -> [Int]? {
        if let obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
            if let arrNums = obj["shape"] as? [NSNumber], arrNums.count == 4 {
                return arrNums.map { $0.intValue }
            }
            if let arrInts = obj["shape"] as? [Int], arrInts.count == 4 {
                return arrInts
            }
        }
        return nil
    }

    func processRawTensorFromDocuments(basename: String) {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("❌ Documents directory not found")
            return
        }
        let dir = docs.appendingPathComponent("DebugModelInputs", isDirectory: true)
        let binURL = dir.appendingPathComponent(basename).appendingPathExtension("bin")
        let jsonURL = dir.appendingPathComponent(basename).appendingPathExtension("json")
        do {
            let data = try Data(contentsOf: binURL)
            // Try to read meta, but default to [1,3,256,256]
            var shape: [Int] = [1, 3, 256, 256]
            if let metaData = try? Data(contentsOf: jsonURL),
               let obj = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
               let s = obj["shape"] as? [NSNumber], s.count == 4 {
                shape = s.map { $0.intValue }
            }
            guard let array = try makeInputArrayFromRawFloat32(data: data, shape: shape) else {
                print("❌ Failed to create MLMultiArray from raw tensor")
                return
            }
            // Debug: stats from MLMultiArray
            if let channelStats = computeAllChannelStats(from: array) {
                let fmt: (Float) -> String = { String(format: "%.4f", $0) }
                if channelStats.count >= 3 {
                    let r = channelStats[0]; let g = channelStats[1]; let b = channelStats[2]
                    print("RAW Tensor Channel stats (pre-norm 0-255): R(mean=\(fmt(r.mean)), std=\(fmt(r.std))) G(mean=\(fmt(g.mean)), std=\(fmt(g.std))) B(mean=\(fmt(b.mean)), std=\(fmt(b.std)))")
                } else {
                    print("RAW Tensor Channel stats: \(channelStats.map{ "(mean=\(fmt($0.mean)), std=\(fmt($0.std)))" }.joined(separator: ", ")))" )
                }
            }
            // Predict
            try predict(with: array, sourceTag: "RAW Tensor")
        } catch {
            print("❌ Could not read raw tensor: \(error)")
        }
    }

    private func makeInputArrayFromRawFloat32(data: Data, shape: [Int]) throws -> MLMultiArray? {
        guard shape.count == 4 else { return nil }
        let expectedCount = shape.reduce(1, *)
        let expectedBytes = expectedCount * MemoryLayout<Float>.size
        guard data.count == expectedBytes else {
            print("⚠️ Raw tensor size mismatch: bytes=\(data.count) expected=\(expectedBytes)")
            return nil
        }
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        data.withUnsafeBytes { rawBuf in
            let src = rawBuf.bindMemory(to: Float.self)
            let dst = array.dataPointer.bindMemory(to: Float.self, capacity: expectedCount)
            dst.assign(from: src.baseAddress!, count: expectedCount)
        }
        return array
    }

    private func predict(with inputArray: MLMultiArray, sourceTag: String) throws {
        guard let model = model else { return }
        let input = try MLDictionaryFeatureProvider(dictionary: [self.inputFeatureName: inputArray])
        let output = try model.prediction(from: input)
        let outKey = output.featureNames.contains(self.outputFeatureName)
            ? self.outputFeatureName
            : (model.modelDescription.outputDescriptionsByName.keys.first ?? self.outputFeatureName)
        if let outputValue = output.featureValue(for: outKey), let multiArray = outputValue.multiArrayValue {
            let noActionScore = Float(truncating: multiArray[0])
            let actionScore = Float(truncating: multiArray[1])
            // Log probabilities
            print("\(sourceTag) Prediction - Action Score: \(actionScore), No Action Score: \(noActionScore)")
            // Derive stable log-probabilities and log-odds (difference of logits) from softmax outputs
            let eps: Float = 1e-12
            let p0 = max(min(noActionScore, 1 - eps), eps)
            let p1 = max(min(actionScore, 1 - eps), eps)
            let logp0 = log(Double(p0))
            let logp1 = log(Double(p1))
            let logOdds = logp1 - logp0 // equals (logit1 - logit0)
            let meanLogP = (logp0 + logp1) / 2.0
            let centeredLogit0 = logp0 - meanLogP
            let centeredLogit1 = logp1 - meanLogP
            let fmt: (Double) -> String = { String(format: "%.6f", $0) }
            print("\(sourceTag) Derived logits (centered): no_action=\(fmt(centeredLogit0)), action=\(fmt(centeredLogit1)), log-odds=\(fmt(logOdds)))")
            self.evaluateDecision(actionScore: actionScore, noActionScore: noActionScore)
        }
    }

    // Run CoreML inference on a preprocessed UIImage (expected already cropped+resized to 256x256)
    func processUIImage(_ uiImage: UIImage) {
        if suppressMainSession { return }
        processUIImage(uiImage, completion: nil)
    }

    // Variant with completion for chaining multiple images sequentially
    func processUIImage(_ uiImage: UIImage, completion: (() -> Void)?) {
        guard isModelLoaded, let model = model else { return }
        if suppressMainSession { return }
        guard let cg = uiImage.cgImage else { return }
        let ci = CIImage(cgImage: cg)

        // Expect preprocessed 256x256 input; update previews directly
        if let small = renderUIImage(from: ci, targetSize: CGSize(width: 160, height: 160)) {
            DispatchQueue.main.async { self.previewImage = small }
        }
        if let big = renderUIImage(from: ci, targetSize: CGSize(width: 300, height: 300)) {
            DispatchQueue.main.async { self.bigPreviewImage = big }
        }

        // Validate size is 256x256, otherwise skip to avoid mismatched tensor shape
        let w = Int(ci.extent.width.rounded())
        let h = Int(ci.extent.height.rounded())
        guard w == 256, h == 256 else {
            print("⚠️ Expected 256x256 preprocessed image, got \(w)x\(h); skipping resize as requested")
            return
        }

        // Convert to pixel buffer
        guard let resizedPixelBuffer = createPixelBuffer(from: ci, width: 256, height: 256) else { return }

        // 5) Pack pixels and predict
        visionQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let inputArray = try self.makeInputArrayFromPixelBuffer(resizedPixelBuffer)
                // Debug: compute per-channel (R,G,B) mean/std (pre-norm, 0-255) from the actual model input tensor
                #if DEBUG
                if let channelStats = self.computeAllChannelStats(from: inputArray) {
                    let fmt: (Float) -> String = { String(format: "%.4f", $0) }
                    if channelStats.count >= 3 {
                        let r = channelStats[0]; let g = channelStats[1]; let b = channelStats[2]
                        print("UIImage Channel stats (pre-norm 0-255): R(mean=\(fmt(r.mean)), std=\(fmt(r.std))) G(mean=\(fmt(g.mean)), std=\(fmt(g.std))) B(mean=\(fmt(b.mean)), std=\(fmt(b.std)))")
                    } else {
                        print("UIImage Channel stats: \(channelStats.map{ "(mean=\(fmt($0.mean)), std=\(fmt($0.std)))" }.joined(separator: ", ")))" )
                    }
                }
                #endif
                let input = try MLDictionaryFeatureProvider(dictionary: [self.inputFeatureName: inputArray])
                let output = try model.prediction(from: input)
                let outKey = output.featureNames.contains(self.outputFeatureName)
                    ? self.outputFeatureName
                    : (model.modelDescription.outputDescriptionsByName.keys.first ?? self.outputFeatureName)
                if let outputValue = output.featureValue(for: outKey), let multiArray = outputValue.multiArrayValue {
                    let noActionScore = Float(truncating: multiArray[0])
                    let actionScore = Float(truncating: multiArray[1])
                    print("UIImage Prediction - Action Score: \(actionScore), No Action Score: \(noActionScore)")
                    self.evaluateDecision(actionScore: actionScore, noActionScore: noActionScore)
                }
                // Signal completion back to caller
                if let completion = completion {
                    DispatchQueue.main.async { completion() }
                }
            } catch {
                print("❌ Prediction error (UIImage): \(error)")
                if let completion = completion {
                    DispatchQueue.main.async { completion() }
                }
            }
        }
    }

    // Compute per-channel (R,G,B) mean/std from a 32BGRA pixel buffer; values in [0,255]
    private func computeChannelStats(from pixelBuffer: CVPixelBuffer) -> (rMean: Float, rStd: Float, gMean: Float, gStd: Float, bMean: Float, bStd: Float)? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let data = baseAddress.assumingMemoryBound(to: UInt8.self)

        var rSum: Double = 0, gSum: Double = 0, bSum: Double = 0
        var rSumSq: Double = 0, gSumSq: Double = 0, bSumSq: Double = 0
        let count = Double(width * height)

        for y in 0..<height {
            let rowOffset = y * bytesPerRow
            for x in 0..<width {
                let p = rowOffset + x * 4
                // BGRA order
                let b = Double(data[p + 0])
                let g = Double(data[p + 1])
                let r = Double(data[p + 2])
                rSum += r; gSum += g; bSum += b
                rSumSq += r * r; gSumSq += g * g; bSumSq += b * b
            }
        }

        let rMean = Float(rSum / count)
        let gMean = Float(gSum / count)
        let bMean = Float(bSum / count)
        let rVar = max(Float(rSumSq / count) - rMean * rMean, 0)
        let gVar = max(Float(gSumSq / count) - gMean * gMean, 0)
        let bVar = max(Float(bSumSq / count) - bMean * bMean, 0)
        return (rMean, rVar.squareRoot(), gMean, gVar.squareRoot(), bMean, bVar.squareRoot())
    }

    // Compute first-channel mean/std from an MLMultiArray [1,3,H,W] or [3,H,W] with Float32 values in [0,255]
    private func computeFirstChannelStats(from array: MLMultiArray) -> (mean: Float, std: Float)? {
        guard array.dataType == .float32 else { return nil }
        let shape = array.shape.map { $0.intValue }
        // Support [1,3,H,W] or [3,H,W]
        var n = 1, c = 3, h = 0, w = 0
        if shape.count == 4 {
            n = shape[0]; c = shape[1]; h = shape[2]; w = shape[3]
        } else if shape.count == 3 {
            c = shape[0]; h = shape[1]; w = shape[2]
        } else {
            return nil
        }
        guard c >= 1, h > 0, w > 0 else { return nil }
        let strides = array.strides.map { $0.intValue }
        let base = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        // Compute stats over the first channel (index 0) of the first batch (index 0)
        let nStride = (shape.count == 4 ? strides[0] : 0)
        let cStride = (shape.count == 4 ? strides[1] : strides[0])
        let hStride = (shape.count == 4 ? strides[2] : strides[1])
        let wStride = (shape.count == 4 ? strides[3] : strides[2])
        let baseOffset = 0 * nStride + 0 * cStride
        var count: Int = 0
        var mean: Double = 0
        var m2: Double = 0
        for yy in 0..<h {
            let rowBase = baseOffset + yy * hStride
            for xx in 0..<w {
                let idx = rowBase + xx * wStride
                let v = Double(base[idx])
                count += 1
                // Welford's online algorithm
                let delta = v - mean
                mean += delta / Double(count)
                let delta2 = v - mean
                m2 += delta * delta2
            }
        }
        if count <= 1 { return (Float(mean), 0) }
        let variance = m2 / Double(count - 1)
        return (Float(mean), Float(variance.squareRoot()))
    }

    // Compute mean/std for every channel (assumes channels packed as R,G,B in CHW) for [1,3,H,W] or [3,H,W]
    private func computeAllChannelStats(from array: MLMultiArray) -> [(mean: Float, std: Float)]? {
        guard array.dataType == .float32 else { return nil }
        let shape = array.shape.map { $0.intValue }
        var n = 1, c = 3, h = 0, w = 0
        if shape.count == 4 {
            n = shape[0]; c = shape[1]; h = shape[2]; w = shape[3]
        } else if shape.count == 3 {
            c = shape[0]; h = shape[1]; w = shape[2]
        } else { return nil }
        guard c >= 1, h > 0, w > 0 else { return nil }
        let strides = array.strides.map { $0.intValue }
        let base = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let nStride = (shape.count == 4 ? strides[0] : 0)
        let cStride = (shape.count == 4 ? strides[1] : strides[0])
        let hStride = (shape.count == 4 ? strides[2] : strides[1])
        let wStride = (shape.count == 4 ? strides[3] : strides[2])
        let batchOffset = 0 * nStride
        var results: [(mean: Float, std: Float)] = []
        results.reserveCapacity(c)
        for ch in 0..<c {
            let channelBase = batchOffset + ch * cStride
            var count: Int = 0
            var mean: Double = 0
            var m2: Double = 0
            for yy in 0..<h {
                let rowBase = channelBase + yy * hStride
                for xx in 0..<w {
                    let idx = rowBase + xx * wStride
                    let v = Double(base[idx])
                    count += 1
                    let delta = v - mean
                    mean += delta / Double(count)
                    let delta2 = v - mean
                    m2 += delta * delta2
                }
            }
            if count <= 1 {
                results.append((Float(mean), 0))
            } else {
                let variance = m2 / Double(count - 1)
                results.append((Float(mean), Float(variance.squareRoot())))
            }
        }
        return results
    }
}
