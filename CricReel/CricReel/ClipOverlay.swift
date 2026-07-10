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

/// Everything needed to caption one delivery's clip.
struct OverlayInfo {
    /// Batting-team score BEFORE this ball (+ the other team), reel names.
    let scoreBefore: String
    /// Batting-team score AFTER this ball (+ the other team), reel names.
    let scoreAfter: String
    /// "3.4  BOWL → BAT"
    let delivery: String
    /// "FOUR" / "Bowled" / "1 run"
    let outcome: String
    /// "4" / "6" / "W" / "1" / "Wd"
    let badge: String
    let accent: OverlayAccent
}

enum ClipOverlay {

    static func info(for d: Delivery, match: Match, lookup: PlayerLookup) -> OverlayInfo {
        let (before, after) = scoreLines(for: d, match: match, lookup: lookup)
        return OverlayInfo(
            scoreBefore: before,
            scoreAfter: after,
            delivery: "\(d.overNumber).\(d.ballInOver)  \(lookup.reel(d.bowlerID)) → \(lookup.reel(d.strikerID))",
            outcome: DeliveryFormatting.outcome(d),
            badge: DeliveryFormatting.badge(d),
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

    /// Batting-team score before/after this delivery, alongside the other team's total.
    static func scoreLines(for d: Delivery, match: Match, lookup: PlayerLookup) -> (before: String, after: String) {
        guard let innings = d.innings else {
            let line = DeliveryFormatting.scoreLine(match: match)
            return (line, line)
        }
        let battingIsA = innings.battingTeamIsA
        var runsBefore = 0, wktsBefore = 0
        for x in innings.orderedDeliveries where x.sequence < d.sequence {
            runsBefore += x.totalRuns
            if x.isWicket { wktsBefore += 1 }
        }
        let runsAfter = runsBefore + d.totalRuns
        let wktsAfter = wktsBefore + (d.isWicket ? 1 : 0)

        let batName = match.teamReelName(isTeamA: battingIsA)
        let other = MatchScoring.teamScore(isTeamA: !battingIsA, in: match)
        let otherName = match.teamReelName(isTeamA: !battingIsA)
        let otherStr = other.batted ? "\(otherName) \(other.runs)/\(other.wickets)" : otherName

        return ("\(batName) \(runsBefore)/\(wktsBefore)   ·   \(otherStr)",
                "\(batName) \(runsAfter)/\(wktsAfter)   ·   \(otherStr)")
    }
}

/// In-app playback caption (static — shows the ball's final score).
struct ClipOverlayView: View {
    let info: OverlayInfo

    var body: some View {
        VStack(spacing: 7) {
            Text(info.scoreAfter)
                .font(.headline.weight(.semibold))
                .foregroundStyle(.white.opacity(0.95))
            Text(info.delivery)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.75))
            HStack(spacing: 10) {
                Text(info.badge)
                    .font(.title3.bold()).monospacedDigit()
                    .frame(minWidth: 40, minHeight: 40)
                    .background(info.accent.color.gradient, in: Circle())
                    .overlay(Circle().strokeBorder(.white.opacity(0.35), lineWidth: 1))
                    .foregroundStyle(.white)
                Text(info.outcome)
                    .font(.title2.weight(.heavy))
                    .foregroundStyle(.white)
            }
        }
        .multilineTextAlignment(.center)
        .padding(.horizontal, 20).padding(.vertical, 14)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .strokeBorder(
                    LinearGradient(colors: [info.accent.color.opacity(0.85),
                                            info.accent.color.opacity(0.25)],
                                   startPoint: .top, endPoint: .bottom),
                    lineWidth: 1.5)
        )
        .shadow(color: .black.opacity(0.35), radius: 8, y: 3)
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
