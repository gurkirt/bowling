# CricReel

Standalone iOS cricket scoring app. Reuses only the bowling-action **trigger + clip-writing** core from the CriClips app; CriClips is left untouched. Adds local match/team/player management, ball-by-ball scoring, clip↔ball linking, commentary + replay, per-player journeys, and auto/on-demand highlight reels.

Design rationale and roadmap: [../docs/plan.md](../docs/plan.md).

## Requirements
- Xcode 15+ on macOS.
- iOS 17+ device (developed against iPhone 14 Plus). Camera features do not work in the Simulator.
- No third-party dependencies (SwiftUI + SwiftData + AVFoundation + CoreML only).

## Build & run
1. Open `CricReel.xcodeproj` in Xcode.
2. Select the `CricReel` scheme and a real device (camera required).
3. Set your signing team if `9K9Q387VZD` is not yours: target **CricReel** → Signing & Capabilities → Team. Do the same for the `CricReelTests` target.
4. Build & run (⌘R).

## Tests
Pure scoring rules are unit-tested (no device needed):
- Run `CricReelTests` with ⌘U (or Product → Test).
- Covers strike rotation, wides, wickets, all-out / overs-limit completion, and undo.

## How it works (quick tour)
- **Players / Teams tabs** — build a local pool and team squads.
- **Matches tab** — create a match (teams, overs, runs-per-wide, playing XI + batting order, toss), then open it.
- **Score Live** — camera + bowling-action detection at top; scoreboard + controls below.
  - *Auto* mode: a detected delivery records a clip and opens a confirm sheet — save (clip links to the ball) or discard a false trigger.
  - *Manual* mode / **Add Ball**: record a delivery with no clip (for missed detections).
  - Detection is armed only when a bowler is selected and the innings is live. Camera + inference stop when the innings completes or you leave the screen.
- **Ball-by-Ball / Scorecard / Highlights** — replay any clip, view derived stats, and build/share reels (auto per-over + on-demand filters for 4s/6s/wickets).

## Storage
- Ball clips: `criclip_<timestamp>.mp4` in the app's Documents directory.
- Highlight reels: `Documents/Highlights/`.
- Match data: on-device SwiftData store.

## Scope (v1)
Runs 0–6, wickets, and wides only. Byes/no-balls/leg-byes/free-hits are intentionally excluded; the data model (`ExtraType` + per-extra run config) is designed to extend later.
