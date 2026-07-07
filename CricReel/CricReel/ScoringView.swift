//
//  ScoringView.swift
//  CricReel
//
//  Live scoring screen. Camera + bowling-action detection at the top; scoreboard and
//  controls below. Detection auto-records a clip and asks the scorer to confirm it
//  (handles false positives); a manual "Add Ball" path handles missed detections.
//
//  Camera capture + model inference are stopped when scoring is done or the screen
//  is dismissed, and resumed on return.
//

import SwiftUI
import SwiftData
import AVFoundation

enum ScoringMode: String, CaseIterable {
    case auto = "Auto"
    case manual = "Manual"
}

struct ScoringView: View {
    @Bindable var match: Match
    @Bindable var innings: Innings

    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \Player.name) private var players: [Player]

    @StateObject private var cameraManager = CameraManager()
    @StateObject private var modelProcessor = ModelProcessor()
    @StateObject private var videoWriter = VideoWriter()

    @State private var mode: ScoringMode = .auto
    @State private var selectedBowlerID: UUID?
    @State private var showingBowlerPicker = false

    // Ball entry state
    @State private var showingEntry = false
    @State private var pendingClipFilename: String?   // set for auto (detected) entries

    private var lookup: PlayerLookup { PlayerLookup(players) }
    private var state: InningsState { MatchScoring.state(for: innings, in: match) }
    private var rules: InningsRules { MatchScoring.rules(for: match, innings: innings) }

    var body: some View {
        VStack(spacing: 0) {
            cameraSection
            scoreboard
            Divider()
            controls
        }
        .navigationTitle("Scoring")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear(perform: setup)
        .onDisappear(perform: pauseEngine)
        .onReceive(NotificationCenter.default.publisher(for: .newClipSaved)) { note in
            handleClipSaved(note)
        }
        .onChange(of: mode) { _, _ in updateDetectionState() }
        .onChange(of: selectedBowlerID) { _, _ in updateDetectionState() }
        .sheet(isPresented: $showingBowlerPicker) {
            BowlerPickerSheet(
                bowlerIDs: MatchScoring.bowlingOrder(for: innings, in: match),
                lookup: lookup,
                current: selectedBowlerID) { selectedBowlerID = $0 }
        }
        .sheet(isPresented: $showingEntry, onDismiss: { pendingClipFilename = nil }) {
            BallEntrySheet(
                clipFilename: pendingClipFilename,
                strikerName: lookup.name(state.strikerID),
                strikerID: state.strikerID,
                nonStrikerName: lookup.name(state.nonStrikerID),
                nonStrikerID: state.nonStrikerID,
                onCommit: { input, keepClip in
                    commit(input, clipFilename: keepClip ? pendingClipFilename : nil)
                    if !keepClip { deletePendingClip() }
                    showingEntry = false
                },
                onDiscard: {
                    deletePendingClip()
                    showingEntry = false
                })
        }
    }

    // MARK: - Camera section

    private var cameraSection: some View {
        ZStack(alignment: .top) {
            if cameraManager.isAuthorized {
                CameraPreview(session: cameraManager.captureSession)
            } else {
                Color.black
                Text("Camera access needed")
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            HStack {
                detectionStatus
                Spacer()
                Picker("Mode", selection: $mode) {
                    ForEach(ScoringMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 160)
            }
            .padding(8)
            .background(.ultraThinMaterial)
        }
        .frame(height: 220)
        .clipped()
    }

    private var detectionStatus: some View {
        Group {
            if mode == .manual {
                Label("Manual", systemImage: "hand.tap")
            } else if videoWriter.isRecording {
                Label("Recording", systemImage: "record.circle").foregroundStyle(.red)
            } else if videoWriter.isCoolingDown {
                Label("Cooldown \(videoWriter.cooldownRemaining)s", systemImage: "hourglass")
            } else if selectedBowlerID == nil {
                Label("Pick bowler", systemImage: "figure.cricket")
            } else if videoWriter.isReadyForTrigger {
                Label("Watching", systemImage: "dot.radiowaves.left.and.right").foregroundStyle(.green)
            } else {
                Label("Buffering", systemImage: "clock")
            }
        }
        .font(.caption).bold()
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(.thinMaterial, in: Capsule())
    }

    // MARK: - Scoreboard

    private var scoreboard: some View {
        let battingName = innings.battingTeamIsA ? match.teamAName : match.teamBName
        return VStack(spacing: 6) {
            HStack {
                Text(battingName).font(.headline)
                Spacer()
                Text("\(state.totalRuns)/\(state.wickets)")
                    .font(.title2).bold().monospacedDigit()
                Text("(\(state.oversDisplay)/\(match.oversPerInnings))")
                    .foregroundStyle(.secondary).monospacedDigit()
            }
            HStack {
                Label(lookup.name(state.strikerID) + " *", systemImage: "figure.cricket")
                Spacer()
                Label(lookup.name(state.nonStrikerID), systemImage: "figure.stand")
            }
            .font(.subheadline)
            HStack {
                Label("Bowler: \(lookup.name(selectedBowlerID))", systemImage: "figure.australian.football")
                Spacer()
            }
            .font(.subheadline).foregroundStyle(.secondary)
        }
        .padding()
    }

    // MARK: - Controls

    @ViewBuilder
    private var controls: some View {
        if state.isInningsComplete {
            completionControls
        } else if selectedBowlerID == nil {
            VStack(spacing: 12) {
                Text(state.oversCompleted == 0 && state.ballsThisOver == 0
                     ? "Select the opening bowler to begin."
                     : "Over complete. Select the next bowler.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                Button {
                    showingBowlerPicker = true
                } label: {
                    Label("Select Bowler", systemImage: "figure.cricket")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
        } else {
            liveControls
        }
    }

    private var liveControls: some View {
        VStack(spacing: 12) {
            Button {
                pendingClipFilename = nil
                showingEntry = true
            } label: {
                Label("Add Ball", systemImage: "plus.circle.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            HStack {
                Button(role: .destructive) { undoLast() } label: {
                    Label("Undo Last", systemImage: "arrow.uturn.backward")
                }
                .disabled(innings.deliveries.isEmpty)
                Spacer()
                Button { showingBowlerPicker = true } label: {
                    Label("Change Bowler", systemImage: "arrow.triangle.2.circlepath")
                }
            }
            .font(.subheadline)

            if mode == .auto {
                Text("Detection is on — a clip is captured and you'll confirm each delivery. Use Add Ball if a delivery is missed.")
                    .font(.caption).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
    }

    private var completionControls: some View {
        VStack(spacing: 12) {
            Image(systemName: "flag.checkered").font(.largeTitle)
            Text(innings.order == 1 ? "Innings complete." : "Match complete.")
                .font(.headline)
            Button("Done") { dismiss() }
                .buttonStyle(.borderedProminent)
        }
        .padding()
        .onAppear { finalizeInningsIfNeeded() }
    }

    // MARK: - Setup / lifecycle

    private func setup() {
        cameraManager.videoWriter = videoWriter
        videoWriter.startCamera()
        let cfg = RecordingConfiguration.default
        cameraManager.updateConfiguration(cfg)
        videoWriter.updateConfiguration(cfg)

        cameraManager.setFrameHandler { [weak videoWriter, weak modelProcessor] sampleBuffer in
            videoWriter?.addFrame(sampleBuffer)
            modelProcessor?.processFrame(sampleBuffer)
        }
        modelProcessor.onTriggerDetected = { [weak videoWriter] in
            videoWriter?.triggerRecording()
        }
        if cameraManager.isAuthorized && !cameraManager.isSessionRunning {
            cameraManager.startSession()
        }
        updateDetectionState()
    }

    /// Detection is only armed in Auto mode with a bowler chosen and the innings live.
    private func updateDetectionState() {
        let shouldRun = (mode == .auto) && selectedBowlerID != nil && !state.isInningsComplete
        modelProcessor.isRunning = shouldRun
    }

    private func pauseEngine() {
        modelProcessor.isRunning = false
        cameraManager.stopSession()
    }

    // MARK: - Clip handling

    private func handleClipSaved(_ note: Notification) {
        guard mode == .auto, selectedBowlerID != nil, !state.isInningsComplete else { return }
        guard let url = note.object as? URL else { return }
        pendingClipFilename = url.lastPathComponent
        showingEntry = true
    }

    private func deletePendingClip() {
        guard let name = pendingClipFilename else { return }
        try? FileManager.default.removeItem(at: ClipStore.url(forClip: name))
        pendingClipFilename = nil
    }

    // MARK: - Commit / undo

    private func commit(_ input: BallInput, clipFilename: String?) {
        guard let bowlerID = selectedBowlerID,
              let strikerID = state.strikerID,
              let nonStrikerID = state.nonStrikerID else { return }

        let s = state
        let extra = ScoringEngine.resolveExtra(input.extraType, rules: rules)
        let commentary = ScoringEngine.commentary(
            over: s.nextOverNumber, ball: s.nextBallInOver,
            bowler: lookup.name(bowlerID), striker: lookup.name(strikerID),
            input: input, rules: rules)

        let delivery = Delivery(
            sequence: innings.deliveries.count,
            overNumber: s.nextOverNumber,
            ballInOver: s.nextBallInOver,
            strikerID: strikerID,
            nonStrikerID: nonStrikerID,
            bowlerID: bowlerID,
            runsOffBat: input.runsOffBat,
            extraType: input.extraType,
            extraRuns: extra.extraRuns,
            isWicket: input.isWicket,
            dismissalType: input.dismissalType,
            dismissedPlayerID: input.isWicket ? (input.dismissedPlayerID ?? strikerID) : nil,
            isLegalDelivery: extra.isLegal,
            clipFilename: clipFilename,
            commentary: commentary,
            highlightTags: ScoringEngine.highlightTags(for: input))
        context.insert(delivery)
        innings.deliveries.append(delivery)

        let newState = state
        if newState.justCompletedOver {
            HighlightBuilder.shared.buildOverReel(match: match, innings: innings,
                                                  overNumber: s.nextOverNumber)
            selectedBowlerID = nil
        }
        if newState.isInningsComplete {
            finalizeInningsIfNeeded()
        }
        updateDetectionState()
    }

    private func undoLast() {
        guard let last = innings.orderedDeliveries.last else { return }
        if let clip = last.clipFilename {
            try? FileManager.default.removeItem(at: ClipStore.url(forClip: clip))
        }
        context.delete(last)
        updateDetectionState()
    }

    private func finalizeInningsIfNeeded() {
        guard state.isInningsComplete, !innings.isComplete else { return }
        innings.isComplete = true
        if innings.order == 1 {
            let second = Innings(order: 2, battingTeamIsA: !innings.battingTeamIsA)
            context.insert(second)
            second.match = match
        } else {
            match.status = .completed
        }
        pauseEngine()
    }
}
