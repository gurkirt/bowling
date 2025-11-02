//
//  line_lengthTests.swift
//  line&lengthTests
//
//  Created by Jean Daniel Browne on 22.10.2025.
//

import XCTest
import AVFoundation
import Combine
@testable import line_length

final class line_lengthTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testExample() throws {
        // Default template test retained intentionally.
    }

    func testPerformanceExample() throws {
        measure {
            // Default template performance test retained intentionally.
        }
    }

    func testModelDetectsActionFramesInSampleVideo() throws {
        let processor = ModelProcessor()
        var cancellables = Set<AnyCancellable>()

        let loadExpectation = expectation(description: "Model loads")
        processor.$isModelLoaded
            .filter { $0 }
            .first()
            .sink { _ in loadExpectation.fulfill() }
            .store(in: &cancellables)

        wait(for: [loadExpectation], timeout: 5.0)

        guard let videoURL = Bundle(for: Self.self).url(forResource: "IMG_8301", withExtension: "MOV") else {
            XCTFail("Missing test resource IMG_8301.MOV")
            return
        }

        let asset = AVAsset(url: videoURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            XCTFail("Video track not found in IMG_8301.MOV")
            return
        }

        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
    output.alwaysCopiesSampleData = false
    XCTAssertTrue(reader.canAdd(output), "Reader cannot add output for video track")
    reader.add(output)
        XCTAssertTrue(reader.startReading(), "Failed to start reading video: \(String(describing: reader.error))")

        let frameSemaphore = DispatchSemaphore(value: 0)
        let pendingFrameQueue = DispatchQueue(label: "pending.frame.index.queue")
        var pendingFrameIndex: Int?
        var results: [(index: Int, action: Float, noAction: Float)] = []
        var skipInitialScore = true

        processor.$lastActionScore
            .sink { actionScore in
                if skipInitialScore {
                    skipInitialScore = false
                    return
                }

                var indexToRecord: Int?
                pendingFrameQueue.sync {
                    indexToRecord = pendingFrameIndex
                    if indexToRecord != nil {
                        pendingFrameIndex = nil
                    }
                }

                guard let frameIndex = indexToRecord else { return }
                let noActionScore = processor.lastNoActionScore
                results.append((frameIndex, actionScore, noActionScore))
                print("Frame \(frameIndex): action=\(actionScore), no_action=\(noActionScore)")
                frameSemaphore.signal()
            }
            .store(in: &cancellables)

        let processingExpectation = expectation(description: "Processed video frames")
        let processingQueue = DispatchQueue(label: "video.processing.queue")
        let frameLimit = 90
        var timeoutFrame: Int?

        processingQueue.async {
            var frameIndex = 0

            while frameIndex < frameLimit, reader.status == .reading,
                  let sampleBuffer = output.copyNextSampleBuffer() {

                pendingFrameQueue.sync {
                    pendingFrameIndex = frameIndex
                }

                processor.processFrame(sampleBuffer)

                let waitResult = frameSemaphore.wait(timeout: .now() + 2.0)
                if waitResult == .timedOut {
                    timeoutFrame = frameIndex
                    break
                }

                frameIndex += 1
            }

            reader.cancelReading()
            processingExpectation.fulfill()
        }

        wait(for: [processingExpectation], timeout: 60.0)

        if let timeoutFrame {
            XCTFail("Timed out waiting for model inference on frame index \(timeoutFrame)")
        }

        XCTAssertFalse(results.isEmpty, "No model outputs captured from sample video")

        let positiveFramesOneBased = results
            .filter { $0.action > $0.noAction }
            .map { $0.index + 1 }

        XCTAssertFalse(positiveFramesOneBased.isEmpty, "Model produced no positive frames")

        let expectedRange = Set(37...44)
        let positives = Set(positiveFramesOneBased)
    XCTAssertTrue(expectedRange.isSubset(of: positives),
              "Expected positive frames 37-44, but positives were \(positiveFramesOneBased)")

        cancellables.removeAll()
    }

}
