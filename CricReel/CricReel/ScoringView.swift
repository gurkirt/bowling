//
//  ScoringView.swift
//  CricReel
//
//  Live scoring: mini scoreboard + model-input preview + CricHeroes-style keypad.
//  Auto-detected clips attach to the next scored ball (discardable). Camera + inference
//  pause when scoring is done or the screen is dismissed.
//

import SwiftUI
import SwiftData

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
    @ObservedObject private var settings = SettingsStore.shared

    @StateObject private var cameraManager = CameraManager()
    @StateObject private var modelProcessor = ModelProcessor()
    @StateObject private var videoWriter = VideoWriter()

    @State private var mode: ScoringMode = .auto
    @State private var selectedBowlerID: UUID?
    @State private var pendingClipFilename: String?

    @State private var showingBowlerPicker = false
    @State private var showingWicket = false
    @State private var wicketPreset: DismissalType?
    @State private var showingSettings = false

    private var lookup: PlayerLookup { PlayerLookup(players) }
    private var state: InningsState { MatchScoring.state(for: innings, in: match) }
    private var target: Int? { MatchScoring.target(for: innings, in: match) }
    private var rules: InningsRules { MatchScoring.rules(for: match, innings: innings) }
    private var battingLines: [UUID: BattingLine] { StatsBuilder.batting(from: innings.orderedDeliveries) }
    private var bowlingLines: [UUID: BowlingLine] { StatsBuilder.bowling(from: innings.orderedDeliveries) }

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                MiniScoreboard(match: match, innings: innings, state: state, target: target,
                               selectedBowlerID: selectedBowlerID, lastBowlerID: lastBowlerID,
                               battingLines: battingLines, bowlingLines: bowlingLines, lookup: lookup)
                previewStrip
                controls
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Scoring")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button { showingSettings = true } label: { Label("Recording Settings", systemImage: "gearshape") }
                    Button(role: .destructive) { endInnings() } label: { Label("End Innings", systemImage: "flag") }
                    Button(role: .destructive) { endMatch() } label: { Label("End Match", systemImage: "flag.checkered") }
                } label: { Image(systemName: "ellipsis.circle") }
            }
        }
        .onAppear(perform: setup)
        .onDisappear(perform: pauseEngine)
        .onReceive(NotificationCenter.default.publisher(for: .newClipSaved)) { handleClipSaved($0) }
        .onChange(of: mode) { _, _ in updateDetectionState() }
        .onChange(of: selectedBowlerID) { _, _ in updateDetectionState() }
        .sheet(isPresented: $showingBowlerPicker) {
            BowlerPickerSheet(bowlerIDs: MatchScoring.bowlingOrder(for: innings, in: match),
                              lookup: lookup, current: selectedBowlerID) { selectedBowlerID = $0 }
        }
        .sheet(isPresented: $showingSettings) { SettingsView() }
        .sheet(isPresented: $showingWicket) { wicketSheet }
    }

    // MARK: - Preview strip

    private var previewStrip: some View {
        HStack(spacing: 12) {
            ZStack {
                if let img = modelProcessor.previewImage {
                    Image(uiImage: img).resizable().scaledToFill()
                } else if cameraManager.isAuthorized {
                    Color.black
                    ProgressView().tint(.white)
                } else {
                    Color.black
                    Text("Camera\nneeded").font(.caption2).foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                }
            }
            .frame(width: 120, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.15)))

            VStack(alignment: .leading, spacing: 8) {
                Picker("Mode", selection: $mode) {
                    ForEach(ScoringMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)

                detectionStatus

                if pendingClipFilename != nil {
                    HStack(spacing: 6) {
                        Image(systemName: "film.fill").foregroundStyle(.green)
                        Text("Clip ready").font(.caption).bold()
                        Spacer()
                        Button {
                            deletePendingClip()
                        } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
                    }
                    .padding(.horizontal, 8).padding(.vertical, 5)
                    .background(.green.opacity(0.12), in: Capsule())
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private var detectionStatus: some View {
        Group {
            if mode == .manual {
                Label("Manual scoring", systemImage: "hand.tap")
            } else if selectedBowlerID == nil {
                Label("Select a bowler to arm detection", systemImage: "figure.cricket")
            } else if videoWriter.isRecording {
                Label("Recording delivery…", systemImage: "record.circle").foregroundStyle(.red)
            } else if videoWriter.isCoolingDown {
                Label("Cooldown \(videoWriter.cooldownRemaining)s", systemImage: "hourglass")
            } else if videoWriter.isReadyForTrigger {
                Label("Watching for delivery", systemImage: "dot.radiowaves.left.and.right").foregroundStyle(.green)
            } else {
                Label("Buffering…", systemImage: "clock")
            }
        }
        .font(.caption).foregroundStyle(.secondary)
    }

    // MARK: - Controls

    @ViewBuilder
    private var controls: some View {
        if state.isInningsComplete {
            completionControls
        } else if selectedBowlerID == nil {
            bowlerPrompt
        } else {
            ScoringPad(enabled: true,
                       onScore: { extra, runs in scoreRuns(extra: extra, padRuns: runs) },
                       onWicket: { preset in openWicket(preset) },
                       onUndo: undoLast,
                       onSwap: swapStrike)
        }
    }

    private var bowlerPrompt: some View {
        VStack(spacing: 12) {
            Text(state.oversCompleted == 0 && state.ballsThisOver == 0
                 ? "Select the opening bowler to begin."
                 : "Over complete. Select the next bowler.")
                .multilineTextAlignment(.center).foregroundStyle(.secondary)
            Button { showingBowlerPicker = true } label: {
                Label("Select Bowler", systemImage: "figure.cricket").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
        }
        .padding().background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private var completionControls: some View {
        VStack(spacing: 12) {
            Image(systemName: "flag.checkered").font(.largeTitle)
            Text(resultText).font(.headline).multilineTextAlignment(.center)
            if innings.order == 1 {
                Button("Start Second Innings") { dismiss() }.buttonStyle(.borderedProminent)
            } else {
                Button("Done") { dismiss() }.buttonStyle(.borderedProminent)
            }
        }
        .padding().frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .onAppear { finalizeInningsIfNeeded() }
    }

    private var resultText: String {
        if innings.order == 2, let t = target {
            let battingName = innings.battingTeamIsA ? match.teamAName : match.teamBName
            if state.totalRuns >= t { return "\(battingName) won!" }
            if state.isAllOut || state.oversCompleted >= match.oversPerInnings {
                let by = (t - 1) - state.totalRuns
                let other = innings.battingTeamIsA ? match.teamBName : match.teamAName
                if by > 0 { return "\(other) won by \(by) run\(by == 1 ? "" : "s")." }
                return "Match tied."
            }
        }
        return innings.order == 1 ? "Innings complete." : "Match complete."
    }

    // MARK: - Wicket

    private var wicketSheet: some View {
        let appeared = MatchScoring.appearedBatters(for: innings, in: match)
        let order = MatchScoring.battingOrder(for: innings, in: match)
        let available = order.filter { !appeared.contains($0) }
        let lastBall = state.ballsThisOver == match.ballsPerOver - 1
        return WicketSheet(
            presetDismissal: wicketPreset,
            strikerID: state.strikerID ?? order.first ?? UUID(),
            nonStrikerID: state.nonStrikerID ?? UUID(),
            strikerName: lookup.name(state.strikerID),
            nonStrikerName: lookup.name(state.nonStrikerID),
            fieldingSideIDs: MatchScoring.bowlingOrder(for: innings, in: match),
            availableBatters: available,
            isLastBallOfOver: lastBall,
            lookup: lookup,
            clipFilename: pendingClipFilename,
            onCommit: { input in commit(input) },
            onCancel: {},
            onDiscardClip: pendingClipFilename != nil ? { deletePendingClip() } : nil)
    }

    private func openWicket(_ preset: DismissalType?) {
        wicketPreset = preset
        showingWicket = true
    }

    // MARK: - Setup / lifecycle

    private func setup() {
        cameraManager.videoWriter = videoWriter
        videoWriter.startCamera()
        applySettings()

        cameraManager.setFrameHandler { [weak videoWriter, weak modelProcessor] sampleBuffer in
            videoWriter?.addFrame(sampleBuffer)
            modelProcessor?.processFrame(sampleBuffer)
        }
        modelProcessor.onTriggerDetected = { [weak videoWriter] in videoWriter?.triggerRecording() }
        if cameraManager.isAuthorized && !cameraManager.isSessionRunning { cameraManager.startSession() }

        // Resume mid-over: default the bowler to whoever bowled the current over.
        let s = state
        if s.ballsThisOver > 0 { selectedBowlerID = currentOverBowlerID }
        updateDetectionState()
    }

    private func applySettings() {
        let cfg = settings.recordingConfiguration
        cameraManager.updateConfiguration(cfg)
        videoWriter.updateConfiguration(cfg)
        modelProcessor.scoreThreshold = settings.scoreThreshold
        modelProcessor.windowSize = settings.windowSize
        modelProcessor.requiredActionCount = settings.requiredActionCount
    }

    private func updateDetectionState() {
        modelProcessor.isRunning = (mode == .auto) && selectedBowlerID != nil && !state.isInningsComplete
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
    }

    private func deletePendingClip() {
        guard let name = pendingClipFilename else { return }
        try? FileManager.default.removeItem(at: ClipStore.url(forClip: name))
        pendingClipFilename = nil
    }

    // MARK: - Commit / undo / swap

    private func scoreRuns(extra: ExtraType, padRuns: Int) {
        var input = BallInput()
        input.extra = extra
        input.padRuns = padRuns
        commit(input)
    }

    private func commit(_ input: BallInput) {
        guard let bowlerID = selectedBowlerID,
              let strikerID = state.strikerID,
              let nonStrikerID = state.nonStrikerID else { return }

        let s = state
        let r = ScoringEngine.resolve(extra: input.extra, padRuns: input.padRuns, rules: rules)
        let commentary = ScoringEngine.commentary(
            over: s.nextOverNumber, ball: s.nextBallInOver,
            bowler: lookup.name(bowlerID), striker: lookup.name(strikerID),
            input: input, resolved: r,
            dismissedName: input.dismissedPlayerID.map { lookup.name($0) },
            fielderName: input.fielderID.map { lookup.name($0) })

        let delivery = Delivery(
            sequence: innings.deliveries.count,
            overNumber: s.nextOverNumber,
            ballInOver: s.nextBallInOver,
            strikerID: strikerID,
            nonStrikerID: nonStrikerID,
            bowlerID: bowlerID,
            runsOffBat: r.runsOffBat,
            extraType: input.extra,
            extraRuns: r.extraRuns,
            physicalRuns: r.physicalRuns,
            bowlerChargedRuns: r.bowlerChargedRuns,
            isLegalDelivery: r.isLegal,
            facedByBatsman: r.faced,
            isWicket: input.isWicket,
            dismissalType: input.dismissal,
            dismissedPlayerID: input.dismissedPlayerID,
            fielderID: input.fielderID,
            newBatterID: input.newBatterID,
            strikerAfterID: input.strikerAfterID,
            nonStrikerAfterID: input.nonStrikerAfterID,
            clipFilename: pendingClipFilename,
            commentary: commentary,
            highlightTags: ScoringEngine.highlightTags(for: input))
        context.insert(delivery)
        innings.deliveries.append(delivery)
        pendingClipFilename = nil

        let newState = state
        if newState.justCompletedOver {
            HighlightBuilder.shared.buildOverReel(match: match, innings: innings, overNumber: s.nextOverNumber)
            selectedBowlerID = nil
        }
        if newState.isInningsComplete { finalizeInningsIfNeeded() }
        updateDetectionState()
    }

    private func undoLast() {
        guard let last = innings.orderedDeliveries.last else { return }
        if let clip = last.clipFilename {
            try? FileManager.default.removeItem(at: ClipStore.url(forClip: clip))
        }
        // Re-open the over's bowler if we undo back into a completed over.
        context.delete(last)
        let s = state
        selectedBowlerID = s.ballsThisOver > 0 ? currentOverBowlerID : nil
        updateDetectionState()
    }

    /// Swap strike by writing an explicit post-state override onto the last delivery.
    private func swapStrike() {
        guard let last = innings.orderedDeliveries.last else { return }
        let s = state
        last.strikerAfterID = s.nonStrikerID
        last.nonStrikerAfterID = s.strikerID
    }

    // MARK: - Innings / match completion

    private func endInnings() {
        innings.isComplete = true
        if innings.order == 1 {
            ensureSecondInnings()
        } else {
            match.status = .completed
        }
        pauseEngine()
        dismiss()
    }

    private func endMatch() {
        innings.isComplete = true
        match.status = .completed
        pauseEngine()
        dismiss()
    }

    private func finalizeInningsIfNeeded() {
        guard state.isInningsComplete, !innings.isComplete else { return }
        innings.isComplete = true
        if innings.order == 1 {
            ensureSecondInnings()
        } else {
            match.status = .completed
        }
        pauseEngine()
    }

    private func ensureSecondInnings() {
        guard !match.innings.contains(where: { $0.order == 2 }) else { return }
        let second = Innings(order: 2, battingTeamIsA: !innings.battingTeamIsA)
        context.insert(second)
        second.match = match
    }

    // MARK: - Derived bowlers

    private var currentOverBowlerID: UUID? {
        let over = state.nextOverNumber
        return innings.orderedDeliveries.last(where: { $0.overNumber == over })?.bowlerID
    }

    private var lastBowlerID: UUID? {
        let over = state.nextOverNumber
        return innings.orderedDeliveries.last(where: { $0.overNumber < over })?.bowlerID
    }
}
