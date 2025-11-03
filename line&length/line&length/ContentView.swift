//
//  ContentView.swift
//  line&length
//
//  Created by Jean Daniel Browne on 22.10.2025.
//

import SwiftUI
import AVFoundation
import AVKit
import UIKit
import Photos

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var videoWriter = VideoWriter()
    @StateObject private var modelProcessor = ModelProcessor()
    
    @State private var showingVideos = false
    @State private var savedVideos: [URL] = []
    @State private var selectedFrameRate: FrameRateOption = RecordingConfiguration.default.frameRate
    @State private var showingVideoReadyBanner = false
    @State private var isAutoTriggerEnabled = true
    // No test-mode branching; demo-focused build
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Camera Preview
                if cameraManager.isAuthorized {
                    CameraPreviewView(cameraManager: cameraManager)
                        .frame(height: 300)
                        .cornerRadius(15)
                        .overlay(
                            RoundedRectangle(cornerRadius: 15)
                                .stroke(Color.blue, lineWidth: 2)
                        )
                } else {
        VStack {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.gray)
                        Text("Camera access required")
                            .font(.headline)
                            .foregroundColor(.gray)
                        Button("Grant Permission") {
                            // This will trigger the permission request
                            cameraManager.checkAuthorization()
                        }
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .frame(height: 300)
                }
                
                // Status Indicators
                VStack(spacing: 10) {
                    HStack {
                        Text("Camera Status:")
                        Spacer()
                        Text(cameraManager.isSessionRunning ? "Running" : "Stopped")
                            .foregroundColor(cameraManager.isSessionRunning ? .green : .red)
                    }
                    
                    HStack {
                        Text("Status:")
                        Spacer()
                        Text(videoWriter.isReadyForTrigger ? "Ready" : "Buffering...")
                            .foregroundColor(videoWriter.isReadyForTrigger ? .green : .orange)
                    }
                    
                    // Frame Statistics
                    if cameraManager.isSessionRunning {
                        HStack {
                            Text("Frames: \(cameraManager.frameCount)")
                                .foregroundColor(.green)
                            Spacer()
                            Text("Dropped: \(cameraManager.droppedFrameCount)")
                                .foregroundColor(cameraManager.droppedFrameCount > 0 ? .orange : .green)
                        }
                        .font(.caption)
                    }

                    if isAutoTriggerEnabled, let uiImage = modelProcessor.previewImage {
                        HStack(spacing: 12) {
                            Image(uiImage: uiImage)
                                .resizable()
                                .frame(width: 80, height: 80)
                                .cornerRadius(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.blue, lineWidth: 1)
                                )
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Auto Trigger Preview")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "Action: %.2f", modelProcessor.lastActionScore))
                                    .font(.caption2)
                                Text(String(format: "No Action: %.2f", modelProcessor.lastNoActionScore))
                                    .font(.caption2)
                            }
                            Spacer()
                        }
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
                
                // Controls
                VStack(spacing: 15) {
                    // Main Trigger Button
                    Button(action: {
                        videoWriter.triggerRecording()
                        // Show banner after 2 seconds to ensure video is fully written
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                            loadSavedVideos() // Refresh the video list
                            showingVideoReadyBanner = true
                            // Hide banner after 3 seconds
                            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                                withAnimation {
                                    showingVideoReadyBanner = false
                                }
                            }
                        }
                    }) {
                        Text(videoWriter.isReadyForTrigger ? "Trigger" : "BUFFERING...")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(videoWriter.isReadyForTrigger ? Color.red : Color.gray)
                            .cornerRadius(15)
                    }
                    .disabled(!cameraManager.isSessionRunning || !videoWriter.isReadyForTrigger)
                    
                    // Secondary Controls
                    HStack(spacing: 12) {
                        Button(action: {
                            loadSavedVideos()
                            showingVideos = true
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "video.fill")
                                Text("Videos")
                            }
                            .padding(.vertical, 10)
                            .padding(.horizontal, 14)
                            .background(Color.purple)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        
                        Button(action: toggleFrameRate) {
                            HStack(spacing: 6) {
                                Text(selectedFrameRate.displayName)
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                                Image(systemName: "arrow.triangle.2.circlepath")
                                    .font(.subheadline)
                            }
                            .padding(.vertical, 10)
                            .padding(.horizontal, 14)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }

                        Button(action: {
                            isAutoTriggerEnabled.toggle()
                            if isAutoTriggerEnabled && selectedFrameRate != .fps60 {
                                selectedFrameRate = .fps60
                                applyRecordingConfiguration()
                            }
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "bolt.badge.automatic")
                                    .font(.subheadline)
                                Text(isAutoTriggerEnabled ? "Auto" : "Manual")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                            }
                            .padding(.vertical, 10)
                            .padding(.horizontal, 14)
                            .background(isAutoTriggerEnabled ? Color.green : Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                    }
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Trigger Recorder")
            .overlay(
                Group {
                    if showingVideoReadyBanner {
                        VStack {
                            HStack(spacing: 12) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("Video Ready!")
                                    .foregroundColor(.white)
                                Spacer()
                                Button("View") {
                                    loadSavedVideos()
                                    showingVideos = true
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 4)
                                .background(Color.blue)
                                .cornerRadius(8)
                            }
                            .padding()
                            .background(Color.black.opacity(0.8))
                            .cornerRadius(12)
                            .padding()
                        }
                        .transition(.move(edge: .top))
                        .animation(.spring(), value: showingVideoReadyBanner)
                        .frame(maxHeight: .infinity, alignment: .top)
                    }
                }
            )
            .onAppear {
                // Set up the connection between CameraManager and VideoWriter
                cameraManager.videoWriter = videoWriter

                // Set up frame handler before starting camera
                setupCameraFrameHandler()
                applyRecordingConfiguration()

                // Start camera and load videos
                videoWriter.startCamera()
                loadSavedVideos()

                // Start camera session if authorized
                if cameraManager.isAuthorized && !cameraManager.isSessionRunning {
                    cameraManager.startSession()
                }

                // Start refresh timer for video list
                startVideoListRefreshTimer()

                // Wire model trigger to writer
                modelProcessor.onTriggerDetected = { [weak videoWriter] in
                    videoWriter?.triggerRecording()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                        loadSavedVideos()
                        showingVideoReadyBanner = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                            withAnimation { showingVideoReadyBanner = false }
                        }
                    }
                }
            }
            .onChange(of: selectedFrameRate) { _ in
                applyRecordingConfiguration()
            }
            .sheet(isPresented: $showingVideos) {
                VideoListView(videos: savedVideos, onVideoDeleted: {
                    loadSavedVideos()
                })
            }
        }
    }
    
    private func toggleFrameRate() {
        selectedFrameRate = selectedFrameRate == .fps30 ? .fps60 : .fps30
    }

    private func applyRecordingConfiguration() {
        let configuration = RecordingConfiguration(resolution: .hd1080,
                                                   frameRate: selectedFrameRate)
        cameraManager.updateConfiguration(configuration)
        videoWriter.updateConfiguration(configuration)
    }
    
    private func setupCameraFrameHandler() {
        cameraManager.setFrameHandler { [weak videoWriter, weak modelProcessor] sampleBuffer in
            // Using weak reference to avoid retain cycles
            videoWriter?.addFrame(sampleBuffer)
            if self.isAutoTriggerEnabled {
                modelProcessor?.processFrame(sampleBuffer)
            }
        }
    }
    
    private func loadSavedVideos() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        
        do {
            let files = try FileManager.default.contentsOfDirectory(at: documentsPath, includingPropertiesForKeys: [.creationDateKey])
            savedVideos = files.filter { $0.pathExtension == "mp4" }
                .sorted { url1, url2 in
                    let date1 = try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate
                    let date2 = try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate
                    return (date1 ?? Date.distantPast) > (date2 ?? Date.distantPast)
                }
        } catch {
            print("Error loading videos: \(error)")
        }
    }
    
    // Timer to automatically refresh video list
    private func startVideoListRefreshTimer() {
        Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            if showingVideos {
                loadSavedVideos()
            }
        }
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager
    
    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.setupPreviewLayer(with: cameraManager.captureSession)
        return view
    }
    
    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        uiView.updateFrame()
    }
}

class CameraPreviewUIView: UIView {
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        backgroundColor = .black
    }
    
    func setupPreviewLayer(with session: AVCaptureSession) {
        // Remove existing preview layer if any
        previewLayer?.removeFromSuperlayer()
        
        // Create new preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.frame = bounds
        
        // Set the video rotation for iOS 17 and later
        if #available(iOS 17.0, *) {
            if let connection = previewLayer?.connection,
               connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90  // 90 degrees for portrait
            }
        } else {
            if let connection = previewLayer?.connection,
               connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }
        
        if let previewLayer = previewLayer {
            layer.addSublayer(previewLayer)
            #if DEBUG
            print("Preview layer added with frame: \(previewLayer.frame)")
            #endif
        }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        updateFrame()
    }
    
    func updateFrame() {
        previewLayer?.frame = bounds
        #if DEBUG
        print("Preview layer frame updated to: \(bounds)")
        #endif
    }
}

struct VideoListView: View {
    @State private var videos: [URL]
    @State private var selectedVideos = Set<URL>()
    @State private var isEditing = false
    @Environment(\.presentationMode) var presentationMode
    let onVideoDeleted: () -> Void
    
    init(videos: [URL], onVideoDeleted: @escaping () -> Void) {
        self._videos = State(initialValue: videos)
        self.onVideoDeleted = onVideoDeleted
    }
    
    private func deleteVideos(_ videosToDelete: [URL]) {
        for videoURL in videosToDelete {
            do {
                try FileManager.default.removeItem(at: videoURL)
                if let index = videos.firstIndex(of: videoURL) {
                    videos.remove(at: index)
                }
                onVideoDeleted()
            } catch {
                print("Error deleting video \(videoURL.lastPathComponent): \(error)")
            }
        }
        selectedVideos.removeAll()
    }
    
    var body: some View {
        NavigationView {
            List(videos, id: \.self, selection: $selectedVideos) { videoURL in
                VideoRowView(videoURL: videoURL, 
                           allVideos: videos, 
                             onDelete: {
                    deleteVideos([videoURL])
                }, isEditing: isEditing)
                .listRowInsets(EdgeInsets())
                .listRowSeparator(.hidden)
            }
            .listStyle(PlainListStyle())
            .environment(\.editMode, .constant(isEditing ? .active : .inactive))
            .navigationTitle("Saved Videos (\(videos.count))")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(isEditing ? "Cancel" : "Edit") {
                        isEditing.toggle()
                        if !isEditing {
                            selectedVideos.removeAll()
                        }
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    if isEditing {
                        Button("Delete (\(selectedVideos.count))") {
                            deleteVideos(Array(selectedVideos))
                            isEditing = false
                        }
                        .disabled(selectedVideos.isEmpty)
                        .foregroundColor(.red)
                    } else {
                        Button("Done") {
                            presentationMode.wrappedValue.dismiss()
                        }
                    }
                }
            }
        }
    }
}

struct VideoRowView: View {
    let videoURL: URL
    let allVideos: [URL]
    let onDelete: () -> Void
    let isEditing: Bool
    @State private var showingPlayer = false
    @State private var showingDeleteAlert = false
    @State private var fileSize: String = "Unknown"
    @State private var creationDate: String = "Unknown"
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(videoURL.lastPathComponent)
                        .font(.headline)
                        .lineLimit(1)
                    
                    HStack {
                        Text("📅 \(creationDate)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Text("📦 \(fileSize)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                // Save and Delete buttons only
                HStack(spacing: 16) {
                    Image(systemName: "photo.badge.plus")
                        .font(.title3)
                        .foregroundColor(.green)
                        .onTapGesture {
                            print("Save button tapped for video: \(videoURL.lastPathComponent)")
                            saveToPhotos()
                        }
                    
                    Image(systemName: "trash.circle.fill")
                        .font(.title3)
                        .foregroundColor(.red)
                        .onTapGesture {
                            print("Delete button tapped for video: \(videoURL.lastPathComponent)")
                            showingDeleteAlert = true
                        }
                }
            }
        }
        .onTapGesture {
            print("Row tapped to play video: \(videoURL.lastPathComponent)")
            openVideoInSystemPlayer()
        }
        .padding(.vertical, 4)
        .onAppear {
            loadFileInfo()
        }
        // Removed custom video player - now using system player
        .alert("Delete Video", isPresented: $showingDeleteAlert) {
            Button("Delete", role: .destructive) {
                deleteVideo()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Are you sure you want to delete this video? This action cannot be undone.")
        }
    }
    
    private func loadFileInfo() {
        // Get file size
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: videoURL.path)
            if let fileSizeValue = attributes[.size] as? Int64 {
                fileSize = ByteCountFormatter.string(fromByteCount: fileSizeValue, countStyle: .file)
            }
            if let date = attributes[.creationDate] as? Date {
                let formatter = DateFormatter()
                formatter.dateStyle = .short
                formatter.timeStyle = .short
                creationDate = formatter.string(from: date)
            }
        } catch {
            print("Error getting file info: \(error)")
        }
    }
    
    private func openVideoInSystemPlayer() {
        print("Opening video in system player: \(videoURL.path)")
        print("Video file exists: \(FileManager.default.fileExists(atPath: videoURL.path))")
        
        // Use AVPlayerViewController with swipe support
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first,
           let rootViewController = window.rootViewController {
            
            print("Found root view controller, checking for existing presentations")
            
            // Find the topmost presented view controller
            var topViewController = rootViewController
            while let presentedVC = topViewController.presentedViewController {
                topViewController = presentedVC
            }
            
            // If we're in a sheet, dismiss it first, then present the video player
            if topViewController != rootViewController {
                print("Dismissing current presentation before showing video player")
                topViewController.dismiss(animated: true) {
                    self.presentVideoPlayerWithSwipe(from: rootViewController)
                }
            } else {
                presentVideoPlayerWithSwipe(from: rootViewController)
            }
        } else {
            print("ERROR: Could not find root view controller for presenting video player")
        }
    }
    
    private func presentVideoPlayerWithSwipe(from viewController: UIViewController) {
        print("Presenting AVPlayerViewController with swipe support")
        
        // Find current video index
        guard let currentIndex = allVideos.firstIndex(of: videoURL) else {
            print("ERROR: Could not find current video in list")
            return
        }
        
        // Create a custom view controller that supports swipe
        let swipePlayerViewController = SwipeVideoPlayerViewController(
            videos: allVideos,
            currentIndex: currentIndex
        )
        
        // Present the swipe-enabled player
        viewController.present(swipePlayerViewController, animated: true)
    }
    
    private func saveToPhotos() {
        print("Save to Photos button tapped for video: \(videoURL.lastPathComponent)")
        // Check current authorization status first
        let currentStatus = PHPhotoLibrary.authorizationStatus()
        print("Photo library authorization status: \(currentStatus.rawValue)")
        
        switch currentStatus {
        case .authorized, .limited:
            print("Photo library access already authorized, saving video")
            performSaveToPhotos()
        case .notDetermined:
            print("Photo library access not determined, requesting permission")
            // Request permission
            PHPhotoLibrary.requestAuthorization { status in
                DispatchQueue.main.async {
                    print("Photo library permission request result: \(status.rawValue)")
                    if status == .authorized || status == .limited {
                        self.performSaveToPhotos()
                    } else {
                        print("Photo library access denied by user")
                    }
                }
            }
        case .denied, .restricted:
            print("Photo library access denied or restricted")
        @unknown default:
            print("Unknown photo library authorization status: \(currentStatus.rawValue)")
        }
    }
    
    private func performSaveToPhotos() {
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: self.videoURL)
        }) { success, error in
            DispatchQueue.main.async {
                if success {
                    print("Video saved to Photos library")
                } else {
                    print("Error saving video to Photos: \(error?.localizedDescription ?? "Unknown error")")
                }
            }
        }
    }
    
    private func deleteVideo() {
        do {
            try FileManager.default.removeItem(at: videoURL)
            print("Video deleted: \(videoURL.lastPathComponent)")
            onDelete()
        } catch {
            print("Error deleting video: \(error)")
        }
    }
}

// Swipe-enabled video player view controller
class SwipeVideoPlayerViewController: UIViewController {
    private let videos: [URL]
    private var currentIndex: Int
    private var pageViewController: UIPageViewController!
    private var playerViewControllers: [AVPlayerViewController] = []
    
    init(videos: [URL], currentIndex: Int) {
        self.videos = videos
        self.currentIndex = currentIndex
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupPageViewController()
        setupNavigationBar()
    }
    
    private func setupPageViewController() {
        // Create page view controller
        pageViewController = UIPageViewController(
            transitionStyle: .scroll,
            navigationOrientation: .horizontal,
            options: nil
        )
        pageViewController.dataSource = self
        pageViewController.delegate = self
        
        // Create player view controllers for each video
        for videoURL in videos {
            let player = AVPlayer(url: videoURL)
            let playerViewController = AVPlayerViewController()
            playerViewController.player = player
            playerViewController.showsPlaybackControls = true
            playerViewControllers.append(playerViewController)
        }
        
        // Set initial view controller
        if currentIndex < playerViewControllers.count {
            pageViewController.setViewControllers(
                [playerViewControllers[currentIndex]],
                direction: .forward,
                animated: false,
                completion: nil
            )
            playerViewControllers[currentIndex].player?.play()
        }
        
        // Add page view controller to this view controller
        addChild(pageViewController)
        view.addSubview(pageViewController.view)
        pageViewController.view.frame = view.bounds
        pageViewController.didMove(toParent: self)
    }
    
    private func setupNavigationBar() {
        // Add close button
        navigationItem.leftBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .done,
            target: self,
            action: #selector(closeButtonTapped)
        )
        
        // Add title with current video info
        updateTitle()
    }
    
    @objc private func closeButtonTapped() {
        dismiss(animated: true)
    }
    
    private func updateTitle() {
        if currentIndex < videos.count {
            let videoName = videos[currentIndex].lastPathComponent
            navigationItem.title = "\(currentIndex + 1) of \(videos.count) - \(videoName)"
        }
    }
}

// MARK: - UIPageViewControllerDataSource
extension SwipeVideoPlayerViewController: UIPageViewControllerDataSource {
    func pageViewController(_ pageViewController: UIPageViewController, viewControllerBefore viewController: UIViewController) -> UIViewController? {
        guard let currentVC = viewController as? AVPlayerViewController,
              let currentIndex = playerViewControllers.firstIndex(of: currentVC) else {
            return nil
        }
        
        let previousIndex = currentIndex - 1
        guard previousIndex >= 0 else { return nil }
        
        return playerViewControllers[previousIndex]
    }
    
    func pageViewController(_ pageViewController: UIPageViewController, viewControllerAfter viewController: UIViewController) -> UIViewController? {
        guard let currentVC = viewController as? AVPlayerViewController,
              let currentIndex = playerViewControllers.firstIndex(of: currentVC) else {
            return nil
        }
        
        let nextIndex = currentIndex + 1
        guard nextIndex < playerViewControllers.count else { return nil }
        
        return playerViewControllers[nextIndex]
    }
}

// MARK: - UIPageViewControllerDelegate
extension SwipeVideoPlayerViewController: UIPageViewControllerDelegate {
    func pageViewController(_ pageViewController: UIPageViewController, didFinishAnimating finished: Bool, previousViewControllers: [UIViewController], transitionCompleted completed: Bool) {
        guard completed,
              let currentVC = pageViewController.viewControllers?.first as? AVPlayerViewController,
              let newIndex = playerViewControllers.firstIndex(of: currentVC) else {
            return
        }
        
        // Update current index
        currentIndex = newIndex
        
        // Stop previous video and start new one
        for (index, playerVC) in playerViewControllers.enumerated() {
            if index == currentIndex {
                playerVC.player?.play()
            } else {
                playerVC.player?.pause()
            }
        }
        
        // Update title
        updateTitle()
    }
}


// Removed custom video player - now using QuickLook system player

#Preview {
    ContentView()
}
