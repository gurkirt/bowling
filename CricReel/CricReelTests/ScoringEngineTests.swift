//
//  ScoringEngineTests.swift
//  CricReelTests
//
//  Unit tests for the pure scoring rules: strike rotation, wides, wickets, undo/replay.
//

import XCTest
@testable import CricReel

final class ScoringEngineTests: XCTestCase {

    // Four batters so we don't hit "all out" during these tests.
    let p1 = UUID(), p2 = UUID(), p3 = UUID(), p4 = UUID()
    var order: [UUID] { [p1, p2, p3, p4] }

    func rules(overs: Int = 6, ballsPerOver: Int = 6, runsPerWide: Int = 1, wideLegal: Bool = false) -> InningsRules {
        InningsRules(oversLimit: overs, ballsPerOver: ballsPerOver,
                     runsPerWide: runsPerWide, wideIsLegalBall: wideLegal, lineupSize: 4)
    }

    /// Helper to append a delivery to a mutable list, using the current derived state
    /// for striker/non-striker/over/ball — mirrors how the live UI builds deliveries.
    private func appended(_ list: [DeliveryData], _ input: BallInput, rules r: InningsRules) -> [DeliveryData] {
        var list = list
        let state = ScoringEngine.computeState(battingOrder: order, deliveries: list, rules: r)
        let extra = ScoringEngine.resolveExtra(input.extraType, rules: r)
        let d = DeliveryData(
            sequence: list.count,
            strikerID: state.strikerID ?? order[0],
            nonStrikerID: state.nonStrikerID ?? order[1],
            bowlerID: UUID(),
            runsOffBat: input.runsOffBat,
            extraType: input.extraType,
            extraRuns: extra.extraRuns,
            isWicket: input.isWicket,
            dismissalType: input.dismissalType,
            dismissedPlayerID: input.dismissedPlayerID,
            isLegalDelivery: extra.isLegal)
        list.append(d)
        return list
    }

    // MARK: - Strike rotation

    func testOddRunsRotateStrike() {
        let r = rules()
        var d: [DeliveryData] = []
        d = appended(d, BallInput(runsOffBat: 1), rules: r)
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.strikerID, p2, "1 run should put p2 on strike")
        XCTAssertEqual(s.nonStrikerID, p1)
        XCTAssertEqual(s.totalRuns, 1)
        XCTAssertEqual(s.legalBalls, 1)
    }

    func testEvenRunsKeepStrike() {
        let r = rules()
        var d: [DeliveryData] = []
        d = appended(d, BallInput(runsOffBat: 2), rules: r)
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.strikerID, p1, "2 runs should keep p1 on strike")
        XCTAssertEqual(s.totalRuns, 2)
    }

    func testEndOfOverSwapsStrike() {
        let r = rules(ballsPerOver: 6)
        var d: [DeliveryData] = []
        // 6 dot balls → over complete → strike swaps.
        for _ in 0..<6 { d = appended(d, BallInput(runsOffBat: 0), rules: r) }
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.oversCompleted, 1)
        XCTAssertEqual(s.ballsThisOver, 0)
        XCTAssertEqual(s.strikerID, p2, "end of over should rotate strike")
        XCTAssertTrue(s.justCompletedOver)
    }

    func testSingleOnLastBallNetKeepsStrikeForRunner() {
        let r = rules(ballsPerOver: 6)
        var d: [DeliveryData] = []
        for _ in 0..<5 { d = appended(d, BallInput(runsOffBat: 0), rules: r) }
        // 6th ball: 1 run (swap) then over-end (swap) → net same batter keeps strike.
        d = appended(d, BallInput(runsOffBat: 1), rules: r)
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.oversCompleted, 1)
        XCTAssertEqual(s.strikerID, p1, "runner off the last ball retains strike next over")
    }

    // MARK: - Wides

    func testWideAddsRunsAndDoesNotAdvanceOver() {
        let r = rules(runsPerWide: 1)
        var d: [DeliveryData] = []
        d = appended(d, BallInput(extraType: .wide), rules: r)
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.totalRuns, 1)
        XCTAssertEqual(s.legalBalls, 0, "a wide is not a legal ball")
        XCTAssertEqual(s.ballsThisOver, 0)
        XCTAssertEqual(s.strikerID, p1, "a wide does not rotate strike")
    }

    func testWideRunsPerWideConfigurable() {
        let r = rules(runsPerWide: 5)
        var d: [DeliveryData] = []
        d = appended(d, BallInput(extraType: .wide), rules: r)
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.totalRuns, 5)
        XCTAssertEqual(s.legalBalls, 0)
    }

    func testOverNeedsSixLegalBallsWithWides() {
        let r = rules(ballsPerOver: 6, runsPerWide: 1)
        var d: [DeliveryData] = []
        d = appended(d, BallInput(extraType: .wide), rules: r)   // not legal
        for _ in 0..<6 { d = appended(d, BallInput(runsOffBat: 0), rules: r) } // 6 legal
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.oversCompleted, 1)
        XCTAssertEqual(s.legalBalls, 6)
        XCTAssertEqual(s.totalRuns, 1)
    }

    // MARK: - Wickets

    func testWicketBringsInNextBatterOnStrike() {
        let r = rules()
        var d: [DeliveryData] = []
        d = appended(d, BallInput(isWicket: true, dismissalType: .bowled), rules: r)
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.wickets, 1)
        XCTAssertEqual(s.strikerID, p3, "next batter (p3) comes in on strike")
        XCTAssertEqual(s.nonStrikerID, p2)
        XCTAssertFalse(s.isInningsComplete)
    }

    func testAllOutCompletesInnings() {
        let r = rules()  // lineupSize 4 → all out at 3 wickets
        var d: [DeliveryData] = []
        for _ in 0..<3 { d = appended(d, BallInput(isWicket: true, dismissalType: .bowled), rules: r) }
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.wickets, 3)
        XCTAssertTrue(s.isAllOut)
        XCTAssertTrue(s.isInningsComplete)
    }

    // MARK: - Overs limit

    func testOversLimitCompletesInnings() {
        let r = rules(overs: 1, ballsPerOver: 6)
        var d: [DeliveryData] = []
        for _ in 0..<6 { d = appended(d, BallInput(runsOffBat: 0), rules: r) }
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: d, rules: r)
        XCTAssertEqual(s.oversCompleted, 1)
        XCTAssertTrue(s.isInningsComplete)
    }

    // MARK: - Undo / replay consistency

    func testUndoIsJustDroppingLastDelivery() {
        let r = rules()
        var d: [DeliveryData] = []
        d = appended(d, BallInput(runsOffBat: 4), rules: r)
        d = appended(d, BallInput(runsOffBat: 1), rules: r)
        let before = ScoringEngine.computeState(battingOrder: order, deliveries: Array(d.dropLast()), rules: r)
        XCTAssertEqual(before.totalRuns, 4)
        XCTAssertEqual(before.strikerID, p1)
    }

    // MARK: - Commentary & highlights

    func testHighlightTags() {
        XCTAssertEqual(ScoringEngine.highlightTags(for: BallInput(runsOffBat: 4)), [.four])
        XCTAssertEqual(ScoringEngine.highlightTags(for: BallInput(runsOffBat: 6)), [.six])
        XCTAssertEqual(ScoringEngine.highlightTags(for: BallInput(isWicket: true, dismissalType: .bowled)), [.wicket])
        XCTAssertTrue(ScoringEngine.highlightTags(for: BallInput(runsOffBat: 2)).isEmpty)
    }
}
