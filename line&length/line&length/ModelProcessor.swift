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

class ModelProcessor: ObservableObject {
    @Published var isModelLoaded = false
    @Published var currentPrediction: Bool = false
    @Published var confidence: Float = 0.0
    
    private var model: MLModel?
    private var request: VNCoreMLRequest?
    private let visionQueue = DispatchQueue(label: "model.vision.queue")
    
    // Trigger detection
    private var consecutiveTrueCount = 0
    private let triggerThreshold = 4
    
    var onTriggerDetected: (() -> Void)?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        // TODO: Replace with actual model file when provided by user
        // For now, we'll create a placeholder structure
        
        // Example of how to load a CoreML model:
        /*
        guard let modelURL = Bundle.main.url(forResource: "YourModelName", withExtension: "mlpackage") else {
            print("Model file not found")
            return
        }
        
        do {
            model = try MLModel(contentsOf: modelURL)
            
            // Create Vision request
            request = VNCoreMLRequest(model: try VNCoreMLModel(for: model!)) { [weak self] request, error in
                self?.handlePrediction(request: request, error: error)
            }
            request?.imageCropAndScaleOption = .centerCrop
            
            DispatchQueue.main.async {
                self.isModelLoaded = true
            }
        } catch {
            print("Error loading model: \(error)")
        }
        */
        
        // Placeholder - simulate model loaded
        DispatchQueue.main.async {
            self.isModelLoaded = true
        }
    }
    
    func processFrame(_ sampleBuffer: CMSampleBuffer) {
        guard isModelLoaded else { return }
        
        visionQueue.async { [weak self] in
            guard let self = self else { return }
            
            // For now, simulate processing with random results
            // TODO: Replace with actual CoreML processing
            let simulatedPrediction = Bool.random()
            let simulatedConfidence = Float.random(in: 0.5...1.0)
            
            DispatchQueue.main.async {
                self.currentPrediction = simulatedPrediction
                self.confidence = simulatedConfidence
                
                // Check for trigger condition
                if simulatedPrediction {
                    self.consecutiveTrueCount += 1
                    if self.consecutiveTrueCount >= self.triggerThreshold {
                        self.onTriggerDetected?()
                        self.consecutiveTrueCount = 0 // Reset after trigger
                    }
                } else {
                    self.consecutiveTrueCount = 0
                }
            }
        }
    }
    
    private func handlePrediction(request: VNRequest, error: Error?) {
        guard error == nil,
              let results = request.results as? [VNClassificationObservation],
              let topResult = results.first else {
            return
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            // Assuming the model outputs a classification with "positive" class
            // Adjust based on your actual model's output classes
            let prediction = topResult.identifier == "positive" || topResult.identifier == "true"
            let confidence = topResult.confidence
            
            self.currentPrediction = prediction
            self.confidence = confidence
            
            // Check for trigger condition
            if prediction {
                self.consecutiveTrueCount += 1
                if self.consecutiveTrueCount >= self.triggerThreshold {
                    self.onTriggerDetected?()
                    self.consecutiveTrueCount = 0 // Reset after trigger
                }
            } else {
                self.consecutiveTrueCount = 0
            }
        }
    }
    
    // Method to convert CMSampleBuffer to CIImage for processing
    private func createCIImage(from sampleBuffer: CMSampleBuffer) -> CIImage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }
        return CIImage(cvPixelBuffer: pixelBuffer)
    }
    
    // Reset trigger counter (call after recording is complete)
    func resetTrigger() {
        consecutiveTrueCount = 0
    }
}
