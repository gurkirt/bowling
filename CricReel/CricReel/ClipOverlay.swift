//
//  ClipOverlay.swift
//  CricReel
//
//  Single source of truth for the clip caption overlay: the content model + accent,
//  a SwiftUI view for in-app playback, and a share sheet. The exported (burned-in)
//  version is rendered from the same `OverlayInfo` in HighlightBuilder.
//

import SwiftUI
import UIKit

enum OverlayAccent {
    case four, six, wicket, neutral

    var color: Color {
        switch self {
        case .four:    return .green
        case .six:     return .purple
        case .wicket:  return .red
        case .neutral: return Color(.systemGray)
        }
    }
    var uiColor: UIColor {
        switch self {
        case .four:    return .systemGreen
        case .six:     return .systemPurple
        case .wicket:  return .systemRed
        case .neutral: return .systemGray
        }
    }
}

/// Everything needed to caption one delivery's clip. The score strip renders as white
/// chips: [accent bar][team][score][overs] … [opponent][opponent score]. Score and overs
/// both flip from "before" to "after" at the result reveal.
struct OverlayInfo {
    /// Batting team reel name, e.g. "KW".
    let battingTeam: String
    /// "12/1" before / after this ball.
    let scoreBefore: String
    let scoreAfter: String
    /// "2.3/20" (overs bowled / innings overs) before / after this ball.
    let oversBefore: String
    let oversAfter: String
    /// Opponent reel name, e.g. "ZMA".
    let opponent: String
    /// Opponent's total in the chase ("120/6"); nil in the first innings.
    let opponentScore: String?
    /// "Bumrah → Kohli" — shown until the reveal.
    let delivery: String
    /// Big centre reveal: "4" / "6" / "•" / "Wd2" for runs, dismissal name for wickets.
    let result: String
    let accent: OverlayAccent
}

enum ClipOverlay {

    static func info(for d: Delivery, match: Match, lookup: PlayerLookup) -> OverlayInfo {
        let score = scoreParts(for: d, match: match)
        return OverlayInfo(
            battingTeam: score.team,
            scoreBefore: score.before,
            scoreAfter: score.after,
            oversBefore: score.oversBefore,
            oversAfter: score.oversAfter,
            opponent: score.opponent,
            opponentScore: score.opponentScore,
            delivery: "\(lookup.reel(d.bowlerID)) → \(lookup.reel(d.strikerID))",
            result: d.isWicket ? DeliveryFormatting.outcome(d) : DeliveryFormatting.badge(d),
            accent: accent(for: d))
    }

    static func accent(for d: Delivery) -> OverlayAccent {
        if d.isWicket { return .wicket }
        if d.extraType == .none {
            if d.runsOffBat == 4 { return .four }
            if d.runsOffBat == 6 { return .six }
        }
        return .neutral
    }

    /// Chip contents for the score strip. Overs count only advances for legal
    /// deliveries (a wide/no-ball leaves "2.3/20" unchanged at the reveal).
    static func scoreParts(for d: Delivery, match: Match)
        -> (team: String, before: String, after: String,
            oversBefore: String, oversAfter: String,
            opponent: String, opponentScore: String?) {
        let limit = match.oversPerInnings
        guard let innings = d.innings else {
            let team = match.teamReelName(isTeamA: true)
            return (team, "0/0", "0/0", "0.0/\(limit)", "0.0/\(limit)",
                    match.teamReelName(isTeamA: false), nil)
        }
        let battingIsA = innings.battingTeamIsA
        var runsBefore = 0, wktsBefore = 0
        for x in innings.orderedDeliveries where x.sequence < d.sequence {
            runsBefore += x.totalRuns
            if x.isWicket { wktsBefore += 1 }
        }
        let runsAfter = runsBefore + d.totalRuns
        let wktsAfter = wktsBefore + (d.isWicket ? 1 : 0)

        // d.ballInOver is the (1-based) legal ball being bowled.
        let oversBefore = "\(d.overNumber).\(max(0, d.ballInOver - 1))/\(limit)"
        let oversAfter: String
        if !d.isLegalDelivery {
            oversAfter = oversBefore
        } else if d.ballInOver >= match.ballsPerOver {
            oversAfter = "\(d.overNumber + 1).0/\(limit)"
        } else {
            oversAfter = "\(d.overNumber).\(d.ballInOver)/\(limit)"
        }

        let opponentScore: String?
        if MatchScoring.target(for: innings, in: match) != nil {
            let other = MatchScoring.teamScore(isTeamA: !battingIsA, in: match)
            opponentScore = "\(other.runs)/\(other.wickets)"
        } else {
            opponentScore = nil
        }
        return (match.teamReelName(isTeamA: battingIsA),
                "\(runsBefore)/\(wktsBefore)", "\(runsAfter)/\(wktsAfter)",
                oversBefore, oversAfter,
                match.teamReelName(isTeamA: !battingIsA), opponentScore)
    }
}

/// In-app playback overlay — fills the video frame. White score chips along the top
/// (accent bar turns the result colour at the reveal), "Bowler → Batter" centred until
/// the reveal, then the result in big white letters. Same delayed-reveal style as the
/// burned-in reels, driven by the same setting.
struct ClipOverlayView: View {
    let info: OverlayInfo
    var revealDelay: Double = SettingsStore.shared.overlayRevealDelay

    @State private var revealed = false

    var body: some View {
        GeometryReader { geo in
            ZStack {
                VStack {
                    scoreStrip
                        .padding(.top, 12)
                    Spacer()
                }

                // Delivery / result sit 22% of the way down, clear of the action.
                Group {
                    if revealed {
                        Text(info.result)
                            .font(.system(size: 56, weight: .heavy))
                            .foregroundStyle(resultColor)
                            .shadow(color: .black.opacity(0.9), radius: 12, y: 3)
                            .lineLimit(1).minimumScaleFactor(0.35)
                            .padding(.horizontal, 24)
                            .transition(.scale(scale: 0.6).combined(with: .opacity))
                    } else {
                        Text(info.delivery)
                            .font(.system(size: 22, weight: .semibold))
                            .foregroundStyle(.white)
                            .shadow(color: .black.opacity(0.8), radius: 7, y: 2)
                            .lineLimit(1).minimumScaleFactor(0.5)
                            .padding(.horizontal, 24)
                    }
                }
                .position(x: geo.size.width / 2, y: geo.size.height * 0.22)
            }
        }
        .allowsHitTesting(false)
        .task {
            revealed = false
            try? await Task.sleep(nanoseconds: UInt64(max(0.2, revealDelay) * 1_000_000_000))
            withAnimation(.spring(response: 0.35, dampingFraction: 0.7)) { revealed = true }
        }
    }

    /// Reveal text takes the accent colour (white for plain runs/dots).
    private var resultColor: Color {
        info.accent == .neutral ? .white : info.accent.color
    }

    /// Two rows each side, inset toward the middle: team name (bigger) on top,
    /// score + overs boxes beneath it; opponent mirrored on the right.
    private var scoreStrip: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 5) {
                chip(info.battingTeam, font: .title3.weight(.bold))
                HStack(spacing: 5) {
                    chip(revealed ? info.scoreAfter : info.scoreBefore, font: .callout.weight(.semibold))
                    chip(revealed ? info.oversAfter : info.oversBefore, font: .callout.weight(.semibold))
                }
            }
            Spacer(minLength: 10)
            VStack(alignment: .trailing, spacing: 5) {
                chip(info.opponent, font: .title3.weight(.bold))
                if let s = info.opponentScore { chip(s, font: .callout.weight(.semibold)) }
            }
        }
        .padding(.horizontal, 28)
    }

    private func chip(_ text: String, font: Font) -> some View {
        Text(text)
            .font(font).monospacedDigit()
            .foregroundStyle(.black)
            .lineLimit(1)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.white, in: RoundedRectangle(cornerRadius: 7))
    }
}

/// UIActivityViewController wrapper for sharing a captioned clip URL.
struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
