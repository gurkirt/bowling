//
//  ScoringEngineTests.swift
//  CricReelTests
//
//  Unit tests for the scoring rules: extras, strike rotation, wickets, target, undo.
//

import XCTest
@testable import CricReel

final class ScoringEngineTests: XCTestCase {

    let p1 = UUID(), p2 = UUID(), p3 = UUID(), p4 = UUID()
    var order: [UUID] { [p1, p2, p3, p4] }

    func rules(overs: Int = 6, ballsPerOver: Int = 6, runsPerWide: Int = 1,
               runsPerNoBall: Int = 1, lineup: Int = 4, target: Int? = nil) -> InningsRules {
        InningsRules(oversLimit: overs, ballsPerOver: ballsPerOver,
                     runsPerWide: runsPerWide, runsPerNoBall: runsPerNoBall,
                     wideIsLegalBall: false, noBallIsLegalBall: false,
                     lineupSize: lineup, target: target)
    }

    /// Append a normal (non-wicket) delivery using the current derived state.
    private func ball(_ list: [DeliveryData], extra: ExtraType = .none, runs: Int = 0,
                      rules r: InningsRules) -> [DeliveryData] {
        var list = list
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: list, rules: r)
        let res = ScoringEngine.resolve(extra: extra, padRuns: runs, rules: r)
        list.append(DeliveryData(
            sequence: list.count,
            strikerID: s.strikerID ?? p1, nonStrikerID: s.nonStrikerID ?? p2, bowlerID: UUID(),
            runsOffBat: res.runsOffBat, extraType: extra, extraRuns: res.extraRuns,
            physicalRuns: res.physicalRuns, isLegalDelivery: res.isLegal, isWicket: false,
            strikerAfterID: nil, nonStrikerAfterID: nil, newBatterID: nil))
        return list
    }

    /// Append a wicket with explicit post-state (as the WicketSheet would produce).
    private func wicket(_ list: [DeliveryData], newBatter: UUID?, strikerAfter: UUID?,
                        nonStrikerAfter: UUID?, rules r: InningsRules) -> [DeliveryData] {
        var list = list
        let s = ScoringEngine.computeState(battingOrder: order, deliveries: list, rules: r)
        list.append(DeliveryData(
            sequence: list.count,
            strikerID: s.strikerID ?? p1, nonStrikerID: s.nonStrikerID ?? p2, bowlerID: UUID(),
            runsOffBat: 0, extraType: .none, extraRuns: 0, physicalRuns: 0,
            isLegalDelivery: true, isWicket: true,
            strikerAfterID: strikerAfter, nonStrikerAfterID: nonStrikerAfter, newBatterID: newBatter))
        return list
    }

    private func state(_ list: [DeliveryData], _ r: InningsRules) -> InningsState {
        ScoringEngine.computeState(battingOrder: order, deliveries: list, rules: r)
    }

    // MARK: - Strike rotation

    func testOddRunsRotateStrike() {
        let r = rules()
        let s = state(ball([], runs: 1, rules: r), r)
        XCTAssertEqual(s.strikerID, p2)
        XCTAssertEqual(s.totalRuns, 1)
        XCTAssertEqual(s.legalBalls, 1)
    }

    func testEvenRunsKeepStrike() {
        let r = rules()
        let s = state(ball([], runs: 2, rules: r), r)
        XCTAssertEqual(s.strikerID, p1)
        XCTAssertEqual(s.totalRuns, 2)
    }

    func testEndOfOverSwapsStrike() {
        let r = rules(ballsPerOver: 6)
        var d: [DeliveryData] = []
        for _ in 0..<6 { d = ball(d, runs: 0, rules: r) }
        let s = state(d, r)
        XCTAssertEqual(s.oversCompleted, 1)
        XCTAssertEqual(s.ballsThisOver, 0)
        XCTAssertEqual(s.strikerID, p2, "dot ball to end the over rotates strike")
        XCTAssertTrue(s.justCompletedOver)
    }

    func testSingleOffLastBallRetainsStrike() {
        let r = rules(ballsPerOver: 6)
        var d: [DeliveryData] = []
        for _ in 0..<5 { d = ball(d, runs: 0, rules: r) }
        d = ball(d, runs: 1, rules: r)  // single (swap) then over-end (swap) = net same
        let s = state(d, r)
        XCTAssertEqual(s.oversCompleted, 1)
        XCTAssertEqual(s.strikerID, p1)
    }

    // MARK: - Extras

    func testWideAddsRunsAndDoesNotAdvanceOver() {
        let r = rules(runsPerWide: 1)
        let s = state(ball([], extra: .wide, runs: 0, rules: r), r)
        XCTAssertEqual(s.totalRuns, 1)
        XCTAssertEqual(s.legalBalls, 0)
        XCTAssertEqual(s.strikerID, p1)
    }

    func testWidePlusRunsRotatesStrike() {
        let r = rules(runsPerWide: 1)
        let s = state(ball([], extra: .wide, runs: 1, rules: r), r)  // 1 wide + 1 bye run
        XCTAssertEqual(s.totalRuns, 2)
        XCTAssertEqual(s.legalBalls, 0)
        XCTAssertEqual(s.strikerID, p2)
    }

    func testNoBallOffBat() {
        let r = rules(runsPerNoBall: 1)
        let res = ScoringEngine.resolve(extra: .noBall, padRuns: 4, rules: r)
        XCTAssertEqual(res.runsOffBat, 4)
        XCTAssertEqual(res.extraRuns, 1)
        XCTAssertEqual(res.bowlerChargedRuns, 5)
        XCTAssertFalse(res.isLegal)
        XCTAssertTrue(res.faced)
        let s = state(ball([], extra: .noBall, runs: 4, rules: r), r)
        XCTAssertEqual(s.totalRuns, 5)
        XCTAssertEqual(s.legalBalls, 0)
    }

    func testByeIsLegalNotChargedToBowler() {
        let r = rules()
        let res = ScoringEngine.resolve(extra: .bye, padRuns: 2, rules: r)
        XCTAssertEqual(res.extraRuns, 2)
        XCTAssertEqual(res.bowlerChargedRuns, 0)
        XCTAssertTrue(res.isLegal)
        let s = state(ball([], extra: .bye, runs: 2, rules: r), r)
        XCTAssertEqual(s.totalRuns, 2)
        XCTAssertEqual(s.legalBalls, 1)
        XCTAssertEqual(s.strikerID, p1)
    }

    func testOverNeedsSixLegalBallsWithExtras() {
        let r = rules(ballsPerOver: 6)
        var d = ball([], extra: .wide, runs: 0, rules: r)
        d = ball(d, extra: .noBall, runs: 0, rules: r)
        for _ in 0..<6 { d = ball(d, runs: 0, rules: r) }
        let s = state(d, r)
        XCTAssertEqual(s.oversCompleted, 1)
        XCTAssertEqual(s.legalBalls, 6)
        XCTAssertEqual(s.totalRuns, 2)
    }

    // MARK: - Wickets

    func testWicketOverridePlacesNewBatter() {
        let r = rules()
        let d = wicket([], newBatter: p3, strikerAfter: p3, nonStrikerAfter: p2, rules: r)
        let s = state(d, r)
        XCTAssertEqual(s.wickets, 1)
        XCTAssertEqual(s.strikerID, p3)
        XCTAssertEqual(s.nonStrikerID, p2)
        XCTAssertFalse(s.isInningsComplete)
    }

    func testAllOutCompletesInnings() {
        let r = rules(lineup: 4)
        var d = wicket([], newBatter: p3, strikerAfter: p3, nonStrikerAfter: p2, rules: r)
        d = wicket(d, newBatter: p4, strikerAfter: p4, nonStrikerAfter: p2, rules: r)
        d = wicket(d, newBatter: nil, strikerAfter: nil, nonStrikerAfter: nil, rules: r)
        let s = state(d, r)
        XCTAssertEqual(s.wickets, 3)
        XCTAssertTrue(s.isAllOut)
        XCTAssertTrue(s.isInningsComplete)
    }

    func testAppearedBatters() {
        let r = rules()
        let d = wicket([], newBatter: p3, strikerAfter: p3, nonStrikerAfter: p2, rules: r)
        let appeared = ScoringEngine.appearedBatters(battingOrder: order, deliveries: d)
        XCTAssertEqual(appeared, Set([p1, p2, p3]))
    }

    // MARK: - Overs limit & target

    func testOversLimitCompletesInnings() {
        let r = rules(overs: 1, ballsPerOver: 6)
        var d: [DeliveryData] = []
        for _ in 0..<6 { d = ball(d, runs: 0, rules: r) }
        XCTAssertTrue(state(d, r).isInningsComplete)
    }

    func testTargetReachedCompletesInnings() {
        let r = rules(overs: 6, target: 5)
        var d = ball([], runs: 4, rules: r)
        XCTAssertFalse(state(d, r).isInningsComplete)
        d = ball(d, runs: 1, rules: r) // total 5 == target
        let s = state(d, r)
        XCTAssertTrue(s.targetReached)
        XCTAssertTrue(s.isInningsComplete)
    }

    // MARK: - Undo & highlights

    func testUndoIsDroppingLastDelivery() {
        let r = rules()
        var d = ball([], runs: 4, rules: r)
        d = ball(d, runs: 1, rules: r)
        let before = state(Array(d.dropLast()), r)
        XCTAssertEqual(before.totalRuns, 4)
        XCTAssertEqual(before.strikerID, p1)
    }

    func testHighlightTags() {
        XCTAssertEqual(ScoringEngine.highlightTags(for: BallInput(extra: .none, padRuns: 4)), [.four])
        XCTAssertEqual(ScoringEngine.highlightTags(for: BallInput(extra: .none, padRuns: 6)), [.six])
        XCTAssertEqual(ScoringEngine.highlightTags(for: BallInput(isWicket: true, dismissal: .bowled)), [.wicket])
        XCTAssertTrue(ScoringEngine.highlightTags(for: BallInput(extra: .none, padRuns: 2)).isEmpty)
    }
}
