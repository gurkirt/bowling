//
//  ScoringView.swift
//  CricReel
//
//  Main scoring screen. Drives the full innings lifecycle:
//  select openers → select bowler → score → (over ends → next bowler) → innings over
//  (finish/undo) → innings break → next innings. Detection + recording pause while the
//  ball-entry sheet is open or scoring is done.
//

import SwiftUI
import SwiftData

enum ScoringMode: String, CaseIterable {
    case auto = "Auto"
    case manual = "Manual"
}

struct ScoringView: View {
    @Bindable var match: Match

    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \Player.name) private var players: [Player]
    @ObservedObject private var settings = SettingsStore.shared

    @StateObject private var cameraManager = CameraManager()
    @StateObject private var modelProcessor = ModelProcessor()
    @StateObject private var videoWriter = VideoWriter()

    @State private var activeInningsID: UUID?
    @State private var mode: ScoringMode = .auto
    @State private var selectedBowlerID: UUID?
    @State private var pendingClipFilename: String?
    @State private var showingEntry = false
    @State private var showingBowlerPicker = false
    @State private var showingSettings = false

    private var lookup: PlayerLookup { PlayerLookup(players) }

    // Active innings (first incomplete, or the one the user advanced to).
    private var innings: Innings? {
        let sorted = match.innings.sorted { $0.order < $1.order }
        if let id = activeInningsID, let m = sorted.first(where: { $0.id == id }) { return m }
        return sorted.first(where: { !$0.isComplete }) ?? sorted.last
    }

    var body: some View {
        Group {
            if let innings {
                content(for: innings)
            } else {
                ContentUnavailableView("No innings", systemImage: "exclamationmark.triangle")
            }
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Scoring")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button { showingSettings = true } label: { Label("Recording Settings", systemImage: "gearshape") }
                    Button(role: .destructive) { finishInnings() } label: { Label("End Innings", systemImage: "flag") }
                    Button(role: .destructive) { endMatch() } label: { Label("End Match", systemImage: "flag.checkered") }
                } label: { Image(systemName: "ellipsis.circle") }
            }
        }
        .onAppear(perform: setup)
        .onDisappear(perform: pauseEngine)
        .onReceive(NotificationCenter.default.publisher(for: .newClipSaved)) { handleClipSaved($0) }
        .onChange(of: mode) { _, _ in updateDetectionState() }
        .onChange(of: selectedBowlerID) { _, _ in updateDetectionState() }
        .onChange(of: showingEntry) { _, _ in updateDetectionState() }
        .sheet(isPresented: $showingSettings) { SettingsView() }
        .sheet(isPresented: $showingBowlerPicker) { bowlerPickerSheet }
        .sheet(isPresented: $showingEntry) { entrySheet }
    }

    // MARK: - Content per phase

    @ViewBuilder
    private func content(for innings: Innings) -> some View {
        let state = MatchScoring.state(for: innings, in: match)
        ScrollView {
            VStack(spacing: 14) {
                miniScoreboard(innings, state: state)
                switch phase(innings, state) {
                case .selectOpeners:
                    OpenerSelectionView(order: MatchScoring.battingOrder(for: innings, in: match),
                                        lookup: lookup) { striker, nonStriker in
                        innings.openerStrikerID = striker
                        innings.openerNonStrikerID = nonStriker
                        presentBowlerPicker(innings)
                    }
                case .selectBowler:
                    bowlerPrompt(innings, state: state)
                case .scoring:
                    previewStrip
                    scoringControls(innings, state: state)
                case .inningsOver:
                    inningsOverPanel(innings)
                case .inningsBreak:
                    inningsBreakPanel(innings)
                case .matchOver:
                    matchOverPanel(innings, state: state)
                }
            }
            .padding()
        }
    }

    private enum Phase { case selectOpeners, selectBowler, scoring, inningsOver, inningsBreak, matchOver }

    private func phase(_ innings: Innings, _ state: InningsState) -> Phase {
        if innings.isComplete { return innings.order == 1 ? .inningsBreak : .matchOver }
        if state.isInningsComplete { return .inningsOver }
        if MatchScoring.needsOpeners(innings) { return .selectOpeners }
        if selectedBowlerID == nil { return .selectBowler }
        return .scoring
    }

    // MARK: - Mini scoreboard

    private func miniScoreboard(_ innings: Innings, state: InningsState) -> some View {
        MiniScoreboard(match: match, innings: innings, state: state,
                       target: MatchScoring.target(for: innings, in: match),
                       selectedBowlerID: selectedBowlerID, lastBowlerID: lastBowlerID(innings),
                       battingLines: StatsBuilder.batting(from: innings.orderedDeliveries),
                       bowlingLines: StatsBuilder.bowling(from: innings.orderedDeliveries),
                       lookup: lookup)
    }

    // MARK: - Preview strip (auto shows camera, manual shows only the switch)

    private var previewStrip: some View {
        Group {
            if mode == .manual {
                VStack(spacing: 8) {
                    modeSwitch
                    Label("Manual scoring — tap Add Ball for each delivery.", systemImage: "hand.tap")
                        .font(.caption).foregroundStyle(.secondary)
                }
            } else {
                HStack(spacing: 12) {
                    ZStack {
                        if let img = modelProcessor.previewImage {
                            Image(uiImage: img).resizable().scaledToFill()
                        } else {
                            Color.black
                            if cameraManager.isAuthorized { ProgressView().tint(.white) }
                            else { Text("Camera\nneeded").font(.caption2).foregroundStyle(.white).multilineTextAlignment(.center) }
                        }
                    }
                    .frame(width: 110, height: 110)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.15)))

                    VStack(alignment: .leading, spacing: 10) {
                        modeSwitch
                        detectionStatus
                        if pendingClipFilename != nil {
                            HStack(spacing: 6) {
                                Image(systemName: "film.fill").foregroundStyle(.green)
                                Text("Clip ready").font(.caption).bold()
                                Spacer()
                                Button { deletePendingClip() } label: {
                                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                                }
                            }
                            .padding(.horizontal, 8).padding(.vertical, 5)
                            .background(.green.opacity(0.12), in: Capsule())
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var modeSwitch: some View {
        Picker("Mode", selection: $mode) {
            ForEach(ScoringMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
    }

    private var detectionStatus: some View {
        Group {
            if selectedBowlerID == nil {
                Label("Select a bowler to arm detection", systemImage: "baseball")
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

    // MARK: - Scoring controls (Add Ball + equal buttons + this-over strip)

    private func scoringControls(_ innings: Innings, state: InningsState) -> some View {
        VStack(spacing: 12) {
            Button {
                pendingClipFilename = nil
                showingEntry = true
            } label: {
                Label("Add Ball", systemImage: "plus.circle.fill")
                    .font(.title3.bold()).frame(maxWidth: .infinity, minHeight: 54)
            }
            .buttonStyle(.borderedProminent)

            HStack(spacing: 10) {
                equalButton("Undo", systemImage: "arrow.uturn.backward", role: .destructive) { undoLast(innings) }
                    .disabled(innings.deliveries.isEmpty)
                equalButton("Strike", systemImage: "arrow.left.arrow.right") { swapStrike(innings) }
                    .disabled(innings.deliveries.isEmpty)
                equalButton("Bowler", systemImage: "baseball") { showingBowlerPicker = true }
            }

            thisOverStrip(innings, state: state)
        }
    }

    private func equalButton(_ title: String, systemImage: String,
                             role: ButtonRole? = nil, action: @escaping () -> Void) -> some View {
        Button(role: role, action: action) {
            VStack(spacing: 3) {
                Image(systemName: systemImage).font(.body)
                Text(title).font(.caption)
            }
            .frame(maxWidth: .infinity, minHeight: 48)
        }
        .buttonStyle(.bordered)
    }

    private func thisOverStrip(_ innings: Innings, state: InningsState) -> some View {
        let overNo = state.nextOverNumber
        let balls = innings.orderedDeliveries.filter { $0.overNumber == overNo }
        return VStack(alignment: .leading, spacing: 8) {
            Text("This Over \(overNo + 1)").font(.caption).foregroundStyle(.secondary)
            if balls.isEmpty {
                Text("—").font(.headline).foregroundStyle(.tertiary)
            } else {
                FlowLayout(spacing: 8) {
                    ForEach(balls) { ball in
                        Text(DeliveryFormatting.badge(ball))
                            .font(.headline.bold()).monospacedDigit()
                            .frame(minWidth: 38, minHeight: 38)
                            .background(DeliveryFormatting.kind(ball).color, in: Circle())
                            .foregroundStyle(.white)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Phase panels

    private func bowlerPrompt(_ innings: Innings, state: InningsState) -> some View {
        VStack(spacing: 12) {
            Text(state.oversCompleted == 0 && state.ballsThisOver == 0
                 ? "Select the opening bowler."
                 : "Over complete. Select the next bowler.")
                .multilineTextAlignment(.center).foregroundStyle(.secondary)
            Button { showingBowlerPicker = true } label: {
                Label("Select Bowler", systemImage: "baseball").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
        }
        .padding().frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func inningsOverPanel(_ innings: Innings) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "flag.checkered").font(.largeTitle)
            Text(innings.order == 1 ? "Innings complete." : "Chase complete.")
                .font(.headline)
            HStack(spacing: 12) {
                Button(role: .destructive) { undoLast(innings) } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                Button { finishInnings() } label: {
                    Label("Finish", systemImage: "checkmark").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding().frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func inningsBreakPanel(_ innings: Innings) -> some View {
        VStack(spacing: 12) {
            Text("Innings Break").font(.headline)
            Text("First innings done. Start the chase when ready.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            Button { startNextInnings() } label: {
                Label("Start Next Innings", systemImage: "play.fill").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
        }
        .padding().frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func matchOverPanel(_ innings: Innings, state: InningsState) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "trophy.fill").font(.largeTitle).foregroundStyle(.orange)
            Text("Match complete.").font(.headline)
            Button("Done") { dismiss() }.buttonStyle(.borderedProminent)
        }
        .padding().frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Sheets

    private var bowlerPickerSheet: some View {
        let innings = innings
        let eligible = innings.map { MatchScoring.eligibleBowlers(for: $0, in: match, excluding: lastBowlerID($0)) } ?? []
        return BowlerPickerSheet(bowlerIDs: eligible, lookup: lookup, current: selectedBowlerID) {
            selectedBowlerID = $0
        }
    }

    @ViewBuilder
    private var entrySheet: some View {
        if let innings {
            let state = MatchScoring.state(for: innings, in: match)
            let appeared = MatchScoring.appearedBatters(for: innings, in: match)
            let order = MatchScoring.battingOrder(for: innings, in: match)
            let available = order.filter { !appeared.contains($0) }
            let lastBall = state.ballsThisOver == match.ballsPerOver - 1
            ScoringEntryView(
                pendingClipFilename: pendingClipFilename,
                bowlerName: lookup.name(selectedBowlerID),
                strikerID: state.strikerID ?? order.first ?? UUID(),
                nonStrikerID: state.nonStrikerID ?? UUID(),
                strikerName: lookup.name(state.strikerID),
                nonStrikerName: lookup.name(state.nonStrikerID),
                fieldingSideIDs: MatchScoring.bowlingOrder(for: innings, in: match),
                availableBatters: available,
                isLastBallOfOver: lastBall,
                lookup: lookup,
                onCommit: { input in commit(input, into: innings) },
                onDiscardClip: { deletePendingClip() })
        }
    }

    // MARK: - Lifecycle

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
        if let innings {
            let s = MatchScoring.state(for: innings, in: match)
            if s.ballsThisOver > 0 { selectedBowlerID = currentOverBowlerID(innings) }
        }
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
        let live = innings.map { !MatchScoring.state(for: $0, in: match).isInningsComplete } ?? false
        modelProcessor.isRunning = (mode == .auto) && selectedBowlerID != nil && live && !showingEntry
    }

    private func pauseEngine() {
        modelProcessor.isRunning = false
        cameraManager.stopSession()
    }

    private func presentBowlerPicker(_ innings: Innings) {
        selectedBowlerID = nil
        if !MatchScoring.eligibleBowlers(for: innings, in: match, excluding: lastBowlerID(innings)).isEmpty {
            showingBowlerPicker = true
        }
    }

    // MARK: - Clip handling

    private func handleClipSaved(_ note: Notification) {
        let live = innings.map { !MatchScoring.state(for: $0, in: match).isInningsComplete } ?? false
        guard mode == .auto, selectedBowlerID != nil, live, !showingEntry else { return }
        guard let url = note.object as? URL else { return }
        pendingClipFilename = url.lastPathComponent
        showingEntry = true
    }

    private func deletePendingClip() {
        guard let name = pendingClipFilename else { return }
        try? FileManager.default.removeItem(at: ClipStore.url(forClip: name))
        pendingClipFilename = nil
    }

    // MARK: - Commit / undo / swap

    private func commit(_ input: BallInput, into innings: Innings) {
        guard let bowlerID = selectedBowlerID else { return }
        let s = MatchScoring.state(for: innings, in: match)
        guard let strikerID = s.strikerID, let nonStrikerID = s.nonStrikerID else { return }
        let rules = MatchScoring.rules(for: match, innings: innings)
        let r = ScoringEngine.resolve(extra: input.extra, padRuns: input.padRuns, rules: rules)
        let commentary = ScoringEngine.commentary(
            bowler: lookup.name(bowlerID), striker: lookup.name(strikerID),
            input: input, resolved: r,
            dismissedName: input.dismissedPlayerID.map { lookup.name($0) },
            fielderName: input.fielderID.map { lookup.name($0) })

        let delivery = Delivery(
            sequence: innings.deliveries.count,
            overNumber: s.nextOverNumber, ballInOver: s.nextBallInOver,
            strikerID: strikerID, nonStrikerID: nonStrikerID, bowlerID: bowlerID,
            runsOffBat: r.runsOffBat, extraType: input.extra, extraRuns: r.extraRuns,
            physicalRuns: r.physicalRuns, bowlerChargedRuns: r.bowlerChargedRuns,
            isLegalDelivery: r.isLegal, facedByBatsman: r.faced,
            isWicket: input.isWicket, dismissalType: input.dismissal,
            dismissedPlayerID: input.dismissedPlayerID, fielderID: input.fielderID,
            newBatterID: input.newBatterID,
            strikerAfterID: input.strikerAfterID, nonStrikerAfterID: input.nonStrikerAfterID,
            clipFilename: pendingClipFilename, commentary: commentary,
            highlightTags: ScoringEngine.highlightTags(for: input))
        context.insert(delivery)
        innings.deliveries.append(delivery)
        pendingClipFilename = nil

        let newState = MatchScoring.state(for: innings, in: match)
        if newState.justCompletedOver && !newState.isInningsComplete {
            HighlightBuilder.shared.buildOverReel(match: match, innings: innings,
                                                  overNumber: s.nextOverNumber, lookup: lookup)
            presentBowlerPicker(innings)
        }
        updateDetectionState()
    }

    private func undoLast(_ innings: Innings) {
        guard let last = innings.orderedDeliveries.last else { return }
        if let clip = last.clipFilename {
            try? FileManager.default.removeItem(at: ClipStore.url(forClip: clip))
        }
        context.delete(last)
        let s = MatchScoring.state(for: innings, in: match)
        selectedBowlerID = s.ballsThisOver > 0 ? currentOverBowlerID(innings) : nil
        updateDetectionState()
    }

    private func swapStrike(_ innings: Innings) {
        guard let last = innings.orderedDeliveries.last else { return }
        let s = MatchScoring.state(for: innings, in: match)
        last.strikerAfterID = s.nonStrikerID
        last.nonStrikerAfterID = s.strikerID
    }

    // MARK: - Innings / match completion

    private func finishInnings() {
        guard let innings else { return }
        innings.isComplete = true
        activeInningsID = innings.id   // stay on this innings for the break / result panel
        if innings.order == 1 { ensureSecondInnings() } else { match.status = .completed }
        pauseEngine()
    }

    private func startNextInnings() {
        ensureSecondInnings()
        if let second = match.innings.first(where: { $0.order == 2 }) {
            activeInningsID = second.id
            selectedBowlerID = nil
            pendingClipFilename = nil
            cameraManager.startSession()
            updateDetectionState()
        }
    }

    private func endMatch() {
        if let innings { innings.isComplete = true; activeInningsID = innings.id }
        match.status = .completed
        pauseEngine()
    }

    private func ensureSecondInnings() {
        guard !match.innings.contains(where: { $0.order == 2 }), let innings else { return }
        let second = Innings(order: 2, battingTeamIsA: !innings.battingTeamIsA)
        context.insert(second)
        second.match = match
    }

    // MARK: - Derived bowlers

    private func currentOverBowlerID(_ innings: Innings) -> UUID? {
        let over = MatchScoring.state(for: innings, in: match).nextOverNumber
        return innings.orderedDeliveries.last(where: { $0.overNumber == over })?.bowlerID
    }

    private func lastBowlerID(_ innings: Innings) -> UUID? {
        let over = MatchScoring.state(for: innings, in: match).nextOverNumber
        return innings.orderedDeliveries.last(where: { $0.overNumber < over })?.bowlerID
    }
}

// MARK: - Opener selection

private struct OpenerSelectionView: View {
    let order: [UUID]
    let lookup: PlayerLookup
    var onConfirm: (_ striker: UUID, _ nonStriker: UUID) -> Void

    @State private var striker: UUID?
    @State private var nonStriker: UUID?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Opening Batters").font(.headline)
            picker("On strike", selection: $striker, exclude: nonStriker)
            picker("Non-striker", selection: $nonStriker, exclude: striker)
            Button {
                if let s = striker, let n = nonStriker { onConfirm(s, n) }
            } label: {
                Label("Confirm & Select Bowler", systemImage: "arrow.right").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(striker == nil || nonStriker == nil || striker == nonStriker)
        }
        .padding().frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .onAppear {
            if striker == nil { striker = order.first }
            if nonStriker == nil { nonStriker = order.count > 1 ? order[1] : nil }
        }
    }

    private func picker(_ title: String, selection: Binding<UUID?>, exclude: UUID?) -> some View {
        HStack {
            Text(title).foregroundStyle(.secondary)
            Spacer()
            Picker(title, selection: selection) {
                Text("Select").tag(UUID?.none)
                ForEach(order.filter { $0 != exclude }, id: \.self) { id in
                    Text(lookup.name(id)).tag(UUID?.some(id))
                }
            }
            .labelsHidden()
        }
    }
}

// MARK: - Flow layout for the this-over strip

struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            if x + size.width > maxWidth && x > 0 {
                x = 0; y += rowHeight + spacing; rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: proposal.width ?? x, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowHeight: CGFloat = 0
        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX && x > bounds.minX {
                x = bounds.minX; y += rowHeight + spacing; rowHeight = 0
            }
            sub.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
