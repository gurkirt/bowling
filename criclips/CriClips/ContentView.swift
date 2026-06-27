//
//  ContentView.swift
//  CriClips
//
//  Portrait-mode SwiftUI interface.
//  Shows the model's cropped/resized 256×256 input preview (pinch-to-zoom),
//  live action scores, trigger status and access to recorded clips.
//

import SwiftUI
import AVFoundation
import AVKit
import Photos

struct ContentView: View {
    @StateObject private var cameraManager  = CameraManager()
    @StateObject private var videoWriter    = VideoWriter()
    @StateObject private var modelProcessor = ModelProcessor()

    @State private var showingClips = false
    @State private var showingSettings = false
    @State private var showingTestFrames = false
    @State private var savedClips: [URL] = []
    @State private var showClipBanner = false
    // Recording settings
    @State private var selectedFrameRate: FrameRateOption = .fps30
    @State private var selectedResolution: VideoResolution = .hd1080
    @State private var preTriggerDuration: Double = 1.0
    @State private var postTriggerDuration: Double = 2.0
    @State private var cooldownDuration: Double = 10.0
    // Pinch-to-zoom state for crop preview
    @State private var previewScale: CGFloat = 1.0
    @State private var lastPinchScale: CGFloat = 1.0

    var body: some View {
        ZStack(alignment: .top) {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                headerBar
                cropPreviewSection
                    .padding(.top, 12)
                scoreSection
                    .padding(.top, 14)
                statusSection
                    .padding(.top, 10)
                Spacer(minLength: 0)
                bottomBar
                    .padding(.bottom, 32)
            }
            .padding(.horizontal, 16)

            // Clip-saved banner slides in from the top
            if showClipBanner {
                clipSavedBanner
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .padding(.top, 56)
                    .zIndex(10)
            }
        }
        .onAppear(perform: setupApp)
        .sheet(isPresented: $showingClips) {
            ClipsView(clips: savedClips, onDelete: loadClips)
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView(
                preTriggerDuration: $preTriggerDuration,
                postTriggerDuration: $postTriggerDuration,
                cooldownDuration: $cooldownDuration,
                selectedFrameRate: $selectedFrameRate,
                selectedResolution: $selectedResolution,
                modelProcessor: modelProcessor
            )
        }
        .sheet(isPresented: $showingTestFrames) {
            TestFramesView(modelProcessor: modelProcessor)
        }
        .onChange(of: preTriggerDuration)   { _ in applyConfig() }
        .onChange(of: postTriggerDuration)  { _ in applyConfig() }
        .onChange(of: cooldownDuration)     { _ in applyConfig() }
        .onChange(of: selectedFrameRate) { newFPS in
            // 4K + 60 fps is not a valid combination on most devices
            if newFPS == .fps60 && selectedResolution == .uhd4k { selectedResolution = .hd1080 }
            applyConfig()
        }
        .onChange(of: selectedResolution) { newRes in
            if newRes == .uhd4k && selectedFrameRate == .fps60 { selectedFrameRate = .fps30 }
            applyConfig()
        }
        // Pause camera + model whenever any sheet is presented; resume on dismiss
        .onChange(of: showingClips)      { pauseOrResume(presenting: $0) }
        .onChange(of: showingSettings)   { pauseOrResume(presenting: $0) }
        .onChange(of: showingTestFrames) { pauseOrResume(presenting: $0) }
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("CriClips")
                    .font(.title2.bold())
                    .foregroundColor(.white)
                Text("Cricket Action Detector")
                    .font(.caption2)
                    .foregroundColor(Color.white.opacity(0.5))
            }
            Spacer()
            HStack(spacing: 8) {
                Button { showingSettings = true } label: {
                    Image(systemName: "gearshape")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(Color.white.opacity(0.15))
                        .foregroundColor(.white)
                        .cornerRadius(20)
                }
                Button {
                    showingTestFrames = true
                } label: {
                    Image(systemName: "checkmark.shield")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(Color.white.opacity(0.15))
                        .foregroundColor(.white)
                        .cornerRadius(20)
                }
                Button {
                    loadClips()
                    showingClips = true
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "video.fill")
                        Text("\(savedClips.count)")
                            .monospacedDigit()
                    }
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.white.opacity(0.15))
                    .foregroundColor(.white)
                    .cornerRadius(20)
                }
            }
        }
        .padding(.top, 12)
    }

    // MARK: - Crop Preview

    private var cropPreviewSection: some View {
        GeometryReader { geo in
            let size = geo.size.width
            ZStack {
                if let img = modelProcessor.bigPreviewImage {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFill()
                        .frame(width: size, height: size)
                        .clipped()
                        .scaleEffect(previewScale, anchor: .center)
                        .contentShape(Rectangle())
                        .gesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    let delta = value / lastPinchScale
                                    lastPinchScale = value
                                    previewScale = min(max(previewScale * delta, 1.0), 6.0)
                                }
                                .onEnded { _ in lastPinchScale = 1.0 }
                        )
                        .onTapGesture(count: 2) {
                            withAnimation(.spring(response: 0.35)) { previewScale = 1.0 }
                        }
                        .cornerRadius(16)
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(previewBorderColor, lineWidth: 2.5)
                        )
                    // Zoom indicator
                    if previewScale > 1.01 {
                        VStack {
                            Spacer()
                            HStack {
                                Spacer()
                                Text(String(format: "×%.1f", previewScale))
                                    .font(.caption2.monospacedDigit())
                                    .padding(.horizontal, 8).padding(.vertical, 4)
                                    .background(.ultraThinMaterial)
                                    .cornerRadius(8)
                                    .padding(10)
                            }
                        }
                        .frame(width: size, height: size)
                    }
                } else {
                    // Placeholder before first frame arrives
                    ZStack {
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.white.opacity(0.06))
                        if cameraManager.isAuthorized {
                            CameraPreviewView(cameraManager: cameraManager)
                                .cornerRadius(16)
                            VStack(spacing: 8) {
                                Spacer()
                                Text("Starting model…")
                                    .font(.caption)
                                    .foregroundColor(.white.opacity(0.6))
                                    .padding(.bottom, 12)
                            }
                        } else {
                            VStack(spacing: 12) {
                                Image(systemName: "camera.fill")
                                    .font(.system(size: 44))
                                    .foregroundColor(.gray)
                                Text("Camera access required")
                                    .font(.subheadline)
                                    .foregroundColor(.gray)
                                Button("Grant Permission") { cameraManager.checkAuthorization() }
                                    .padding(.horizontal, 20).padding(.vertical, 10)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                    }
                }
            }
            .frame(width: size, height: size)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    /// Border pulses green on active recording, orange during cooldown, blue otherwise.
    private var previewBorderColor: Color {
        if videoWriter.isRecording { return .green }
        if videoWriter.isCoolingDown { return .orange }
        return .blue.opacity(0.7)
    }

    // MARK: - Scores

    private var scoreSection: some View {
        VStack(spacing: 8) {
            ScoreBar(label: "Action",    score: modelProcessor.lastActionScore,   color: .green)
            ScoreBar(label: "No Action", score: modelProcessor.lastNoActionScore, color: .orange)
        }
    }

    // MARK: - Status

    private var statusSection: some View {
        HStack(spacing: 8) {
            // Stop / Start button
            Button {
                if cameraManager.isSessionRunning {
                    cameraManager.stopSession()
                    modelProcessor.isRunning = false
                } else {
                    modelProcessor.isRunning = true
                    cameraManager.startSession()
                }
            } label: {
                Image(systemName: cameraManager.isSessionRunning ? "pause.circle.fill" : "play.circle.fill")
                    .font(.title3)
                    .foregroundColor(cameraManager.isSessionRunning ? .white : .green)
            }

            // Camera status pill
            indicatorPill(
                icon: "camera.fill",
                label: cameraManager.isSessionRunning ? "Live" : "Paused",
                color: cameraManager.isSessionRunning ? .green : .red)

            // Trigger / cooldown pill
            indicatorPill(
                icon: triggerIcon,
                label: triggerLabel,
                color: triggerColor)

            // Cooldown progress (only when cooling down)
            if videoWriter.isCoolingDown {
                Spacer(minLength: 0)
                HStack(spacing: 4) {
                    Image(systemName: "timer").font(.caption2)
                    Text("\(videoWriter.cooldownRemaining)s").font(.caption2.monospacedDigit())
                }
                .foregroundColor(.orange)
                .padding(.horizontal, 8).padding(.vertical, 5)
                .background(Color.orange.opacity(0.18))
                .cornerRadius(8)
            } else {
                Spacer()
                // Frame-rate toggle (quick tap)
                Button {
                    selectedFrameRate = (selectedFrameRate == .fps60) ? .fps30 : .fps60
                    applyConfig()
                } label: {
                    Text(selectedFrameRate.displayName)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(Color.white.opacity(0.12))
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
            }
        }
    }

    private var triggerIcon: String {
        if videoWriter.isRecording  { return "record.circle" }
        if videoWriter.isCoolingDown{ return "timer" }
        return videoWriter.isReadyForTrigger ? "bolt.fill" : "hourglass"
    }

    private var triggerLabel: String {
        if videoWriter.isRecording   { return "Recording" }
        if videoWriter.isCoolingDown { return "Cooldown \(videoWriter.cooldownRemaining)s" }
        return videoWriter.isReadyForTrigger ? "Ready" : "Buffering…"
    }

    private var triggerColor: Color {
        if videoWriter.isRecording   { return .green }
        if videoWriter.isCoolingDown { return .orange }
        return videoWriter.isReadyForTrigger ? .cyan : .gray
    }

    private func indicatorPill(icon: String, label: String, color: Color) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon).font(.caption)
            Text(label).font(.caption.weight(.medium)).monospacedDigit()
        }
        .foregroundColor(color)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(color.opacity(0.12))
        .cornerRadius(8)
    }

    // MARK: - Bottom Bar

    private var bottomBar: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                // Model status
                HStack(spacing: 6) {
                    Circle()
                        .fill(modelProcessor.isModelLoaded ? Color.green : Color.gray)
                        .frame(width: 8, height: 8)
                    Text(modelProcessor.isModelLoaded ? "Model ready" : "Loading model…")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                }
                if modelProcessor.isModelLoaded && !modelProcessor.computeUnitsLabel.isEmpty {
                    Text("•")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.3))
                    // Show configured compute units; “ANE” prefix means Neural Engine is in use
                    Text(modelProcessor.computeUnitsLabel)
                        .font(.caption)
                        .foregroundColor(modelProcessor.computeUnitsLabel.contains("ANE") ? .cyan : .white.opacity(0.5))
                }
                Spacer()
                Text("Double-tap preview to reset zoom")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.3))
            }
        }
    }

    // MARK: - Clip Saved Banner

    private var clipSavedBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
            Text("Clip saved!").foregroundColor(.white).fontWeight(.semibold)
            Spacer()
            Button("View") {
                loadClips()
                showingClips = true
            }
            .foregroundColor(.white)
            .padding(.horizontal, 12).padding(.vertical, 4)
            .background(Color.blue)
            .cornerRadius(8)
        }
        .padding()
        .background(Color.black.opacity(0.85))
        .cornerRadius(12)
        .padding(.horizontal, 16)
    }

    // MARK: - Setup

    /// Pause camera + model inference while a sheet is presented to save resources.
    private func pauseOrResume(presenting: Bool) {
        if presenting {
            modelProcessor.isRunning = false
            cameraManager.stopSession()
        } else {
            // Only resume if no other sheet is still open
            guard !showingClips, !showingSettings, !showingTestFrames else { return }
            cameraManager.startSession()
            modelProcessor.isRunning = true
        }
    }

    private func setupApp() {
        cameraManager.videoWriter = videoWriter
        videoWriter.startCamera()
        applyConfig()
        loadClips()

        // Wire camera frames to VideoWriter and ModelProcessor
        cameraManager.setFrameHandler { [weak videoWriter, weak modelProcessor] sampleBuffer in
            videoWriter?.addFrame(sampleBuffer)
            modelProcessor?.processFrame(sampleBuffer)
        }

        // Wire model trigger to VideoWriter
        modelProcessor.onTriggerDetected = { [weak videoWriter] in
            videoWriter?.triggerRecording()
        }

        // Watch for new clips
        NotificationCenter.default.addObserver(forName: .newClipSaved, object: nil, queue: .main) { [self] _ in
            loadClips()
            showBanner()
        }

        // Start camera session if already authorized
        if cameraManager.isAuthorized && !cameraManager.isSessionRunning {
            cameraManager.startSession()
        }
    }

    private func applyConfig() {
        let cfg = RecordingConfiguration(
            resolution: selectedResolution,
            frameRate: selectedFrameRate,
            preTriggerDuration: preTriggerDuration,
            postTriggerDuration: postTriggerDuration,
            cooldownDuration: cooldownDuration
        )
        cameraManager.updateConfiguration(cfg)
        videoWriter.updateConfiguration(cfg)
    }

    private func loadClips() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let all = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: [.creationDateKey])) ?? []
        savedClips = all
            .filter { $0.pathExtension.lowercased() == "mp4" }
            .sorted {
                let d1 = (try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                let d2 = (try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                return d1 > d2
            }
    }

    private func showBanner() {
        withAnimation(.spring()) { showClipBanner = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            withAnimation { showClipBanner = false }
        }
    }
}

// MARK: - Score Bar

private struct ScoreBar: View {
    let label: String
    let score: Float
    let color: Color

    var body: some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.caption.weight(.medium))
                .foregroundColor(.white.opacity(0.7))
                .frame(width: 70, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.white.opacity(0.1))
                    RoundedRectangle(cornerRadius: 4)
                        .fill(color.opacity(0.8))
                        .frame(width: geo.size.width * CGFloat(score))
                        .animation(.easeOut(duration: 0.1), value: score)
                }
            }
            .frame(height: 10)
            Text(String(format: "%.2f", score))
                .font(.caption2.monospacedDigit())
                .foregroundColor(color)
                .frame(width: 36, alignment: .trailing)
        }
    }
}

// MARK: - Camera Preview (fallback while model warms up)

struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager

    func makeUIView(context: Context) -> _CameraPreviewUIView {
        let v = _CameraPreviewUIView()
        v.setup(session: cameraManager.captureSession)
        return v
    }
    func updateUIView(_ uiView: _CameraPreviewUIView, context: Context) {
        uiView.setNeedsLayout()
    }
}

class _CameraPreviewUIView: UIView {
    private var previewLayer: AVCaptureVideoPreviewLayer?

    func setup(session: AVCaptureSession) {
        backgroundColor = .black
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        if #available(iOS 17.0, *) {
            if let conn = layer.connection, conn.isVideoRotationAngleSupported(90) {
                conn.videoRotationAngle = 90
            }
        } else {
            if let conn = layer.connection, conn.isVideoOrientationSupported {
                conn.videoOrientation = .portrait
            }
        }
        self.layer.addSublayer(layer)
        previewLayer = layer
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}

// MARK: - Clips Library View

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

/// Wrapper so [Any] can drive .sheet(item:)
struct SharePayload: Identifiable {
    let id = UUID()
    let items: [Any]
}

struct ClipsView: View {
    @State private var clips: [URL]
    @State private var selectedClip: URL?
    @State private var thumbnails: [URL: UIImage] = [:]
    @State private var isGridView = true
    // Multi-select
    @State private var isSelecting = false
    @State private var selectedURLs: Set<URL> = []
    // Action feedback
    @State private var sharePayload: SharePayload?
    @State private var photoResultMessage: String?
    @State private var showingPhotoResult = false
    @Environment(\.dismiss) private var dismiss
    let onDelete: () -> Void

    private let gridColumns = [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())]

    init(clips: [URL], onDelete: @escaping () -> Void) {
        _clips = State(initialValue: clips)
        self.onDelete = onDelete
    }

    // MARK: - Body

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                mainContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                if isSelecting {
                    selectionToolbar
                }
            }
            .navigationTitle(isSelecting
                ? (selectedURLs.isEmpty ? "Select Items" : "\(selectedURLs.count) Selected")
                : "Clips (\(clips.count))")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbarItems }
            .sheet(item: $selectedClip) { url in VideoPlayerView(url: url) }
            .sheet(item: $sharePayload) { payload in ShareSheet(items: payload.items) }
            .alert("Photos", isPresented: $showingPhotoResult) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(photoResultMessage ?? "")
            }
            .task { await loadThumbnails() }
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarItems: some ToolbarContent {
        ToolbarItem(placement: .navigationBarLeading) {
            if isSelecting {
                Button("Cancel") {
                    withAnimation { isSelecting = false; selectedURLs.removeAll() }
                }
            } else {
                Button("Done") { dismiss() }
            }
        }
        ToolbarItem(placement: .navigationBarTrailing) {
            HStack(spacing: 16) {
                if !clips.isEmpty {
                    Button(isSelecting ? "Select All" : "Select") {
                        if isSelecting {
                            selectedURLs = Set(clips)
                        } else {
                            withAnimation { isSelecting = true }
                        }
                    }
                    .font(.subheadline)
                }
                Button {
                    withAnimation { isGridView.toggle() }
                } label: {
                    Image(systemName: isGridView ? "list.bullet" : "square.grid.3x3.fill")
                }
            }
        }
    }

    // MARK: - Main Content

    @ViewBuilder
    private var mainContent: some View {
        if clips.isEmpty {
            VStack(spacing: 16) {
                Image(systemName: "video.slash").font(.system(size: 48)).foregroundColor(.secondary)
                Text("No clips yet").foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if isGridView {
            ScrollView {
                LazyVGrid(columns: gridColumns, spacing: 4) {
                    ForEach(clips, id: \.self) { url in gridCell(url) }
                }
                .padding(4)
            }
        } else {
            List {
                ForEach(clips, id: \.self) { url in listRow(url) }
                    .onDelete(perform: isSelecting ? nil : deleteClips)
            }
        }
    }

    // MARK: - Grid Cell

    private func gridCell(_ url: URL) -> some View {
        let isSelected = selectedURLs.contains(url)
        return ZStack(alignment: .topTrailing) {
            ZStack(alignment: .bottomLeading) {
                thumbnailImage(url, size: nil)
                    .aspectRatio(9/16, contentMode: .fill)
                    .clipped()
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(isSelected ? Color.blue : Color.white.opacity(0.1),
                                    lineWidth: isSelected ? 3 : 0.5)
                    )
                if let sz = fileSize(url) {
                    Text(sz)
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(.ultraThinMaterial)
                        .cornerRadius(4)
                        .padding(5)
                }
            }
            // Checkmark overlay in select mode
            if isSelecting {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(isSelected ? Color.white : Color.white, isSelected ? Color.blue : Color.black.opacity(0.4))
                    .shadow(color: .black.opacity(0.4), radius: 2)
                    .padding(6)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if isSelecting { toggleSelection(url) } else { selectedClip = url }
        }
        .contextMenu { cellContextMenu(url) }
    }

    // MARK: - List Row

    private func listRow(_ url: URL) -> some View {
        let isSelected = selectedURLs.contains(url)
        return HStack(spacing: 12) {
            if isSelecting {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(isSelected ? Color.blue : Color.secondary)
                    .onTapGesture { toggleSelection(url) }
            }
            Button {
                if isSelecting { toggleSelection(url) } else { selectedClip = url }
            } label: {
                HStack(spacing: 12) {
                    thumbnailImage(url, size: 56).cornerRadius(6)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(url.lastPathComponent).font(.subheadline).lineLimit(2)
                        if let sz = fileSize(url) {
                            Text(sz).font(.caption).foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                    if !isSelecting {
                        Image(systemName: "play.circle").foregroundColor(.blue)
                    }
                }
            }
            .buttonStyle(.plain)
        }
        .contextMenu { cellContextMenu(url) }
    }

    // MARK: - Context Menu

    @ViewBuilder
    private func cellContextMenu(_ url: URL) -> some View {
        Button { selectedClip = url } label: {
            Label("Play", systemImage: "play.fill")
        }
        Button { shareURLs([url]) } label: {
            Label("Share", systemImage: "square.and.arrow.up")
        }
        Button { saveToPhotos([url]) } label: {
            Label("Save to Photos", systemImage: "photo.on.rectangle.angled")
        }
        Divider()
        Button(role: .destructive) { deleteClip(url) } label: {
            Label("Delete", systemImage: "trash")
        }
    }

    // MARK: - Selection Toolbar (bottom)

    private var selectionToolbar: some View {
        HStack {
            // Save to Photos
            Button {
                saveToPhotos(Array(selectedURLs))
            } label: {
                VStack(spacing: 4) {
                    Image(systemName: "photo.on.rectangle.angled").font(.title3)
                    Text("Save").font(.caption2)
                }
                .frame(maxWidth: .infinity)
            }
            .disabled(selectedURLs.isEmpty)

            Divider().frame(height: 36)

            // Share
            Button {
                shareURLs(Array(selectedURLs))
            } label: {
                VStack(spacing: 4) {
                    Image(systemName: "square.and.arrow.up").font(.title3)
                    Text("Share").font(.caption2)
                }
                .frame(maxWidth: .infinity)
            }
            .disabled(selectedURLs.isEmpty)

            Divider().frame(height: 36)

            // Delete
            Button(role: .destructive) {
                deleteSelected()
            } label: {
                VStack(spacing: 4) {
                    Image(systemName: "trash").font(.title3)
                    Text("Delete").font(.caption2)
                }
                .frame(maxWidth: .infinity)
            }
            .disabled(selectedURLs.isEmpty)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 10)
        .background(.bar)
        .overlay(Divider(), alignment: .top)
    }

    // MARK: - Thumbnail helpers

    @ViewBuilder
    private func thumbnailImage(_ url: URL, size: CGFloat?) -> some View {
        if let img = thumbnails[url] {
            if let size {
                Image(uiImage: img).resizable().scaledToFill()
                    .frame(width: size, height: size).clipped()
            } else {
                Image(uiImage: img).resizable().scaledToFill()
            }
        } else {
            Rectangle()
                .fill(Color(.systemGray5))
                .overlay(Image(systemName: "film").foregroundColor(.secondary))
                .frame(width: size, height: size)
        }
    }

    private func loadThumbnails() async {
        for url in clips where thumbnails[url] == nil {
            if let img = await generateThumbnail(for: url) {
                thumbnails[url] = img
            }
        }
    }

    private func generateThumbnail(for url: URL) async -> UIImage? {
        let asset = AVAsset(url: url)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: 300, height: 300)
        do {
            let (cg, _) = try await gen.image(at: .zero)
            return UIImage(cgImage: cg)
        } catch { return nil }
    }

    // MARK: - Actions

    private func toggleSelection(_ url: URL) {
        if selectedURLs.contains(url) { selectedURLs.remove(url) }
        else { selectedURLs.insert(url) }
    }

    private func shareURLs(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        sharePayload = SharePayload(items: urls)
    }

    private func saveToPhotos(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        var saved = 0
        var failed = 0
        let group = DispatchGroup()
        for url in urls {
            group.enter()
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            } completionHandler: { success, _ in
                if success { saved += 1 } else { failed += 1 }
                group.leave()
            }
        }
        group.notify(queue: .main) {
            if failed == 0 {
                photoResultMessage = saved == 1 ? "1 video saved to Photos." : "\(saved) videos saved to Photos."
            } else {
                photoResultMessage = "\(saved) saved, \(failed) failed. Check Photos permission in Settings."
            }
            showingPhotoResult = true
        }
    }

    private func deleteClips(at offsets: IndexSet) {
        offsets.map { clips[$0] }.forEach { try? FileManager.default.removeItem(at: $0) }
        clips.remove(atOffsets: offsets)
        onDelete()
    }

    private func deleteClip(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
        clips.removeAll { $0 == url }
        onDelete()
    }

    private func deleteSelected() {
        let toDelete = selectedURLs
        toDelete.forEach { try? FileManager.default.removeItem(at: $0) }
        clips.removeAll { toDelete.contains($0) }
        selectedURLs.removeAll()
        isSelecting = false
        onDelete()
    }

    private func fileSize(_ url: URL) -> String? {
        guard let bytes = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64 else { return nil }
        return String(format: "%.1f MB", Double(bytes) / 1_048_576)
    }
}

// MARK: - Video Player

struct VideoPlayerView: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            VideoPlayer(player: AVPlayer(url: url))
                .ignoresSafeArea()
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Done") { dismiss() }
                    }
                }
        }
    }
}

// MARK: - Notification

extension Notification.Name {
    static let newClipSaved = Notification.Name("CriClipsNewClipSaved")
}

// URL: Identifiable for sheet binding
extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}

// MARK: - Settings View

struct SettingsView: View {
    @Binding var preTriggerDuration: Double
    @Binding var postTriggerDuration: Double
    @Binding var cooldownDuration: Double
    @Binding var selectedFrameRate: FrameRateOption
    @Binding var selectedResolution: VideoResolution
    @ObservedObject var modelProcessor: ModelProcessor
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            Form {
                // ── Recording ────────────────────────────────────────────────
                Section("Recording") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Resolution").font(.subheadline).foregroundColor(.secondary)
                        Picker("Resolution", selection: $selectedResolution) {
                            ForEach(VideoResolution.allCases) { res in
                                Text(res.displayName).tag(res)
                            }
                        }
                        .pickerStyle(.segmented)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Frame rate").font(.subheadline).foregroundColor(.secondary)
                        Picker("Frame rate", selection: $selectedFrameRate) {
                            ForEach(FrameRateOption.allCases) { fps in
                                Text(fps.displayName).tag(fps)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(selectedResolution == .uhd4k)
                        if selectedResolution == .uhd4k {
                            Text("4K is limited to 30 fps")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    Stepper(
                        value: $preTriggerDuration,
                        in: 0.25...1.5, step: 0.25
                    ) {
                        LabeledContent("Pre-trigger", value: String(format: "%.2g s", preTriggerDuration))
                    }
                    Stepper(
                        value: $postTriggerDuration,
                        in: 0.25...10.0, step: 0.25
                    ) {
                        LabeledContent("Post-trigger", value: String(format: "%.2g s", postTriggerDuration))
                    }
                    Stepper(
                        value: $cooldownDuration,
                        in: 0.0...120.0, step: 1.0
                    ) {
                        LabeledContent("Cooldown", value: String(format: "%.0f s", cooldownDuration))
                    }
                }
                // ── Model / Trigger ──────────────────────────────────────────
                Section {
                    Stepper(
                        value: $modelProcessor.inferenceFrameInterval,
                        in: 1...8
                    ) {
                        LabeledContent(
                            "Inference interval",
                            value: modelProcessor.inferenceFrameInterval == 1
                                ? "Every frame"
                                : "Every \(modelProcessor.inferenceFrameInterval) frames"
                        )
                    }
                    Stepper(
                        value: $modelProcessor.windowSize,
                        in: 2...20
                    ) {
                        LabeledContent("Window size", value: "\(modelProcessor.windowSize) frames")
                    }
                    Stepper(
                        value: $modelProcessor.requiredActionCount,
                        in: 1...modelProcessor.windowSize
                    ) {
                        LabeledContent("Trigger threshold", value: "\(modelProcessor.requiredActionCount) / \(modelProcessor.windowSize)")
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        LabeledContent("Score threshold", value: String(format: "%.2f", modelProcessor.scoreThreshold))
                        Slider(value: $modelProcessor.scoreThreshold, in: 0.1...0.99, step: 0.05)
                            .accentColor(.blue)
                    }
                } header: {
                    Text("Model & Trigger")
                } footer: {
                    if !modelProcessor.computeUnitsLabel.isEmpty {
                        Label(
                            "Model running on: \(modelProcessor.computeUnitsLabel). \"ANE\" means the Neural Engine (NPU) is active.",
                            systemImage: "cpu"
                        )
                        .font(.caption)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Test Frames View

/// On-device parity check: run bundled test images through the CoreML model
/// and compare against the Python reference scores.
struct TestFramesView: View {
    let modelProcessor: ModelProcessor
    @Environment(\.dismiss) private var dismiss

    // Reference scores recorded from Python (score_debug_frame.py)
    private let frames: [(
        name: String,
        asset: String,
        preprocessing: Bool,
        refNoAction: Float,
        refAction: Float
    )] = [
        ("debug_frame.png",       "debug_frame",      false, 0.026031, 0.974121),
        ("debug_frame_full.png",  "debug_frame_full", true,  0.016739, 0.983398)
    ]

    @State private var scores: [String: (noAction: Float, action: Float)] = [:]
    @State private var running: Set<String> = []

    var body: some View {
        NavigationView {
            List {
                Section {
                    infoRow
                }
                Section("Test Frames") {
                    ForEach(frames, id: \.name) { frame in
                        frameCard(frame)
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Parity Check")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Run All") {
                        for f in frames { runFrame(f) }
                    }
                    .disabled(!modelProcessor.isModelLoaded || !running.isEmpty)
                }
            }
        }
    }

    // ── Info banner ────────────────────────────────────────────────────────────
    private var infoRow: some View {
        HStack(spacing: 10) {
            Image(systemName: "info.circle").foregroundColor(.blue)
            VStack(alignment: .leading, spacing: 2) {
                Text("Python reference scores")
                    .font(.caption.weight(.semibold))
                Text("PyTorch fp16 & CoreML fp16 agree to ≤0.0001 on these frames.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    // ── Per-frame card ─────────────────────────────────────────────────────────
    @ViewBuilder
    private func frameCard(_ frame: (name: String, asset: String, preprocessing: Bool,
                                     refNoAction: Float, refAction: Float)) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header row: thumbnail + title + Run button
            HStack(spacing: 10) {
                if let img = UIImage(named: frame.asset) {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 52, height: 52)
                        .clipped()
                        .cornerRadius(6)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(frame.name).font(.subheadline.weight(.semibold))
                    Text(frame.preprocessing
                         ? "Full frame → crop → 256×256"
                         : "Pre-cropped 256×256 (no crop)")
                        .font(.caption2).foregroundColor(.secondary)
                }
                Spacer()
                Button {
                    runFrame(frame)
                } label: {
                    if running.contains(frame.name) {
                        ProgressView().frame(width: 40)
                    } else {
                        Text("Run")
                            .font(.caption.weight(.semibold))
                            .frame(width: 40)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(!modelProcessor.isModelLoaded || running.contains(frame.name))
            }

            // Scores table
            VStack(spacing: 5) {
                scoreRow(label: "Python ref",
                         noAction: frame.refNoAction, action: frame.refAction,
                         color: .secondary)
                if let s = scores[frame.name] {
                    scoreRow(label: "iOS model",
                             noAction: s.noAction, action: s.action,
                             color: .blue)
                    let delta = abs(s.action - frame.refAction)
                    HStack {
                        Text("Δ action").font(.caption2).foregroundColor(.secondary)
                        Spacer()
                        Text(String(format: "%.6f", delta))
                            .font(.caption2.monospacedDigit())
                            .foregroundColor(delta < 0.02 ? .green : .orange)
                        Image(systemName: delta < 0.02 ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                            .font(.caption2)
                            .foregroundColor(delta < 0.02 ? .green : .orange)
                    }
                } else {
                    Text("Tap Run to score on device")
                        .font(.caption2).foregroundColor(.secondary)
                }
            }
            .padding(10)
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
        .padding(.vertical, 6)
    }

    private func scoreRow(label: String, noAction: Float, action: Float, color: Color) -> some View {
        HStack {
            Text(label).font(.caption).foregroundColor(color).frame(width: 72, alignment: .leading)
            Spacer()
            Text("no_act \(String(format: "%.4f", noAction))")
                .font(.caption2.monospacedDigit()).foregroundColor(.secondary)
            Text("act \(String(format: "%.4f", action))")
                .font(.caption2.monospacedDigit())
                .foregroundColor(action >= 0.5 ? .green : .orange)
                .fontWeight(action >= 0.5 ? .semibold : .regular)
        }
    }

    // ── Inference ──────────────────────────────────────────────────────────────
    private func runFrame(_ frame: (name: String, asset: String, preprocessing: Bool,
                                    refNoAction: Float, refAction: Float)) {
        guard let img = UIImage(named: frame.asset) else {
            print("⚠️ [TestFrames] Asset not found: \(frame.asset)")
            return
        }
        running.insert(frame.name)
        modelProcessor.scoreImage(img, applyPreprocessing: frame.preprocessing) { noAction, action in
            scores[frame.name] = (noAction, action)
            running.remove(frame.name)
            print("📊 [TestFrames] \(frame.name): no_action=\(noAction) action=\(action)")
        }
    }
}
