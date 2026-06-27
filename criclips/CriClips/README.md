# Line & Length - Camera CoreML Video App

## Overview
This iOS/iPad app captures camera frames, processes them through a CoreML model, and automatically records 2.5-second video clips when the model predicts True for 4 consecutive frames.

## Features
- Real-time camera preview
- CoreML model integration with Vision framework
- Automatic video recording on trigger detection
- Circular buffer for 0.5s pre-trigger recording
- H.264 compressed video output
- Video management and playback

## Architecture

### Core Components
1. **CameraManager** - Handles camera capture and frame processing
2. **ModelProcessor** - CoreML model integration with image transforms
3. **FrameBufferManager** - Circular buffer for recent frames
4. **VideoWriter** - Video recording and compression
5. **TriggerController** - Trigger detection logic
6. **ContentView** - Main UI with camera preview and controls

### Trigger Logic
- Model processes each frame and returns boolean prediction
- Triggers when 4 consecutive True predictions are detected
- Records 0.5s before trigger + 2s after trigger = 2.5s total

## Setup Instructions

### 1. Add CoreML Model
1. Add your `.mlpackage` or `.mlmodel` file to the Xcode project
2. Update `ModelProcessor.swift` to load your actual model:
   ```swift
   guard let modelURL = Bundle.main.url(forResource: "YourModelName", withExtension: "mlpackage") else {
       print("Model file not found")
       return
   }
   ```
3. Uncomment and modify the model loading code in `loadModel()` method

### 2. Configure Model Input/Output
- Ensure your model accepts image input (224x224 recommended)
- Modify the Vision request configuration in `ModelProcessor.swift`
- Update the prediction handling based on your model's output format

### 3. Camera Permissions
- Camera permission is already configured in `Info.plist`
- The app will request permission on first launch

## Usage
1. Launch the app and grant camera permission
2. Tap "Start" to begin camera capture
3. The app will show real-time predictions and trigger status
4. When 4 consecutive True predictions occur, video recording starts automatically
5. Access recorded videos via the "Videos" button

## Video Output
- Videos are saved to the Documents directory
- Format: MP4 with H.264 compression
- Resolution: 1920x1080 (or device-dependent)
- Frame rate: 30 fps
- Duration: 2.5 seconds (0.5s before + 2s after trigger)

## Technical Details
- Uses AVFoundation for camera capture and video recording
- Vision framework for image processing and CoreML integration
- Combine framework for reactive programming
- SwiftUI for modern iOS UI

## Requirements
- iOS 14.0+ / iPadOS 14.0+
- Camera access permission
- Device with back camera

## Notes
- The current implementation includes placeholder model processing with simulated predictions
- Replace the simulation code with your actual CoreML model integration
- Video player implementation is basic - enhance as needed for your use case
