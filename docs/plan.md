# Plan: CricReel — Standalone Cricket Scoring App

## Goal
Build **CricReel**, a NEW standalone iOS app (separate Xcode project) that reuses only the bowling-action trigger + clip-writing core from CriClips. CriClips stays untouched. CricReel adds full local cricket scoring: create matches, teams from a local player pool, score ball-by-ball, auto-link trigger clips to each ball (graceful FP/FN), browse commentary + replay any ball, track per-player journeys, and async end-of-over + on-demand highlight reels. Backend/cloud is Phase 2 (design for scale, don't build yet).

## Confirmed decisions
- **Trigger↔score**: TWO modes + graceful FP/FN handling.
  - Auto mode: trigger fires → clip records → confirm sheet ("was this a ball?"). Confirm → enter outcome, clip links to ball. Reject (FP) → discard clip.
  - Manual mode / FN: scorer adds a ball with NO clip. Can optionally attach/re-record a clip later.
- **Persistence**: SwiftData; bump min deployment target to iOS 17.
- **Rules scope v1 (minimal)**: runs 0–6, wickets, wides only. Byes/no-balls/leg-byes/free-hit EXCLUDED but data model designed to extend.
- **Accounts**: on-device Player/Team profiles only, no auth/login.
- **Highlights**: auto short reel at end of each over + on-demand full-match filtered reel.

## Existing architecture (source to copy FROM — CriClips, do not modify)
- `criclips/CriClips/CameraManager.swift` — AVCaptureSession, frame callbacks.
- `criclips/CriClips/ModelProcessor.swift` — CoreML inference, `evaluateDecision()`, `onTriggerDetected`, `isReadyForTrigger`.
- `criclips/CriClips/VideoWriter.swift` — records clip on trigger; emits `Notification.Name.newClipSaved`.
- `criclips/CriClips/FrameBufferManager.swift` — pre/post-trigger circular buffers.
- `criclips/CriClips/RecordingConfiguration.swift` — recording settings.
- `.mlpackage` model bundle (fastvit_sa12_exp21).

## Phase 0 — Scaffold CricReel + copy trigger/clip core (build first)
- Create NEW Xcode project `CricReel` at repo root: `/home/gurkirt/data/gurkirt_codepad/cricshorts/CricReel/CricReel.xcodeproj` (SwiftUI, min iOS 17). Bundle id `com.cricreel.app` (CONFIRMED). Portrait; camera + photos permissions in Info.plist.
- COPY (trigger + clip only, strip debug): `CameraManager.swift`, `VideoWriter.swift`, `FrameBufferManager.swift`, `RecordingConfiguration.swift`, and the `.mlpackage`.
- COPY + TRIM `ModelProcessor.swift`: keep inference + `onTriggerDetected` + `isReadyForTrigger`/cooldown; REMOVE debug preview-image publishing (`previewImage`/`bigPreviewImage`) and score-bar/test hooks.
- DO NOT copy: `ContentView.swift` gallery/settings/test UI, `TestFramesView`, parity testing, pinch-zoom big-preview.
- New minimal `CricReelApp.swift` entry with SwiftData `.modelContainer` (replaces CriClipsApp).
- Verify a bare camera→trigger→clip-saved loop works before layering scoring.

## Phase 1 — Local scoring (build now)

### A. Data model (SwiftData @Model) — new `ScoringModels.swift`
- `Player`: id, name, photoData?, battingStyle?, bowlingStyle?.
- `Team`: id, name, logoData?, players [Player] (pool/squad).
- `Match`: id, date, venue, teamA, teamB, config, tossWinnerTeamId, tossDecision, status(setup/live/completed), innings [Innings].
- `MatchConfig`: oversPerInnings, playersPerSide, ballsPerOver(6), runsPerWide(default 1), wideIsLegalBall(false).
- `Lineup`: match↔team playing XI (subset of Team.players) + batting order.
- `Innings`: battingTeamId, bowlingTeamId, deliveries [Delivery], order (1/2). Totals derived.
- `Delivery` (ball): overNo, ballNoInOver, strikerId, nonStrikerId, bowlerId, runsOffBat, extraType(none/wide), extraRuns, isWicket, dismissalType, dismissedPlayerId?, isLegalDelivery(Bool), clipFilename?, commentary, highlightTags [wicket/four/six], timestamp.
- Per-player match/career stats = DERIVED by aggregating Delivery rows (no denormalized store in v1).

### B. Scoring engine — new `ScoringEngine.swift` (pure Swift, unit-tested)
- Applies a `Delivery` to innings state; computes: total runs, wickets, legal-ball count, over completion, strike rotation.
- Strike rotation: odd runsOffBat → swap; end of legal over → swap.
- Wide: +runsPerWide to extras, NOT a legal ball (over not advanced), no ball faced, no strike change.
- Wicket: +1 wicket, prompt next batsman (from batting order), striker replaced.
- Over/innings completion detection; innings switch.

### C. Match setup + player pool UI — new `PlayerPoolView.swift`, `TeamSetupView.swift`, `MatchSetupView.swift`
- Manage local player pool (add/edit/delete).
- Build two teams, pick playing XI + batting order.
- Match config form: overs, runs-per-wide, toss winner/decision.

### D. Live scoring screen — new `ScoringView.swift` + `BallConfirmSheet.swift`
- Top: camera preview (reuse CameraManager/ModelProcessor) with Auto/Manual mode toggle.
- Middle: live scorecard (score/wickets/overs, striker*, non-striker, bowler).
- Bottom: score pad (0–6, W, Wide, undo).
- Auto mode: on `newClipSaved` → present `BallConfirmSheet` (thumbnail + "confirm ball / discard(FP)"). Confirm → score pad → save Delivery with clipFilename.
- Manual mode: score pad directly → Delivery with no clip (FN). Optional "attach clip".
- Undo/edit last ball.

### E. Ball-by-ball + playback — new `CommentaryView.swift`
- Reverse-chronological list of deliveries with generated commentary text and clip indicator.
- Tap a ball → play its clip (AVPlayer); balls without clips show "no clip".

### F. Player journey/stats — new `PlayerStatsView.swift`
- Per match: batting (runs, balls, 4s, 6s, SR), bowling (overs, runs conceded, wickets, econ).
- Career aggregate across all local matches (derived).

### G. Highlights — new `HighlightBuilder.swift` (AVFoundation)
- `AVMutableComposition` stitches selected clip files.
- Auto: at end of each over, build reel from that over's highlight-tagged deliveries (wicket/4/6); save + surface to user.
- On-demand: `HighlightsView` with filters (wickets / 4s / 6s / combos, whole match) → build + share via share sheet.

### H. Wiring — `CricReelApp.swift`, new `AppRootView.swift`
- SwiftData `ModelContainer` in `CricReelApp`.
- Root navigation: Matches list → (setup | live scoring | scorecard/commentary | highlights | stats). No standalone clip-recorder screen (that stays in CriClips).

## Phase 2 — Backend & scale (design only, don't build)
- Auth + shared player pool (players discoverable across users/teams).
- Cloud sync of matches/scorecards; live scorecard sharing (read-only link/stream).
- Media storage/CDN for clips + reels; server-side reel rendering.
- Suggested stack: managed BaaS (Supabase/Firebase) for fastest path, or custom API (Postgres + object storage) for control. Defer choice until local model proven.
- Keep SwiftData models mappable to server schema (stable UUIDs on all entities).

## Verification
0. Phase 0: bare camera→trigger→clip-saved loop works in CricReel (no scoring layered yet).
1. Xcode build of CricReel succeeds (iOS 17 target, SwiftData container); runs on iPhone 14 Plus.
2. Unit tests in CricReelTests for `ScoringEngine`: strike rotation on odd runs, no rotation on wide, over-end swap, wide adds runsPerWide & doesn't advance over, wicket flow, undo.
3. Manual: full 2-over match end-to-end in Auto mode; verify each clip links to correct ball.
4. Manual: FP path (discard trigger clip) and FN path (manual ball, no clip) both produce correct score.
5. Manual: end-of-over auto reel plays and contains only that over's 4s/6s/wickets.
6. Manual: on-demand "sixes only" reel + share sheet works.
7. Manual: player stats totals reconcile with scorecard.
8. Manual: camera + model inference stop (verify CPU/no preview updates) when scoring completes or user leaves live scoring; resume on return.

## Resolved from review
- Byes/strike-change: keep simple. `extraType` enum + `runsPerX` config kept extensible (hooks only), NO byes/no-balls in v1.
- iOS 17 bump approved; must run on iPhone 14 Plus (iOS 17+, fine).
- Highlight reel rendering runs ASYNC (off scoring path, notify when ready).
- NEW: Pause camera capture + model inference (+ any rendering) on the scoring screen when scoring is done / user leaves live scoring, to save battery/CPU. Resume on return.

### Lifecycle pause (add to Section D + H wiring)
- `ScoringView` owns start/stop of `CameraManager.session` and `ModelProcessor` inference.
- Pause triggers: innings/match marked complete, user navigates away from live scoring, app backgrounded.
- On pause: stop AVCaptureSession, halt frame→model pipeline, cancel/skip in-flight reel renders where safe. Resume on re-entry when scoring still live.
