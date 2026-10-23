//
//  StumpCalibration.swift
//  CricReel
//
//  Where the bowler's-end stumps are in the frame, and the two classifier crops beside them.
//  The fastvit_sa12_s04 model scores one crop either side of the stumps (the right one
//  mirrored), sized in stump heights so the bowler appears at the same scale at any zoom.
//  The stumps don't move during a session, so the user places a box on them once while
//  positioning the camera.
//

import SwiftUI

/// Stump box as fractions of the portrait frame: centre x and base y (bottom of the stumps)
/// of frame width/height, height of frame height, width of the stump height.
struct StumpBox: Codable, Equatable {
    var centreX: CGFloat
    var baseY: CGFloat
    var height: CGFloat
    var widthOverHeight: CGFloat

    /// Median stump box over the app's own training recordings (criclip_*, 1080×1920).
    static let typical = StumpBox(centreX: 0.484, baseY: 0.833, height: 0.182, widthOverHeight: 0.38)

    private static let key = "stumpBox"

    static func load() -> StumpBox {
        guard let data = UserDefaults.standard.data(forKey: key),
              let box = try? JSONDecoder().decode(StumpBox.self, from: data) else { return .typical }
        return box
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) { UserDefaults.standard.set(data, forKey: Self.key) }
    }

    /// Stump rectangle in pixels of a frame of `size`, origin top-left.
    func rect(in size: CGSize) -> CGRect {
        let h = height * size.height
        let w = widthOverHeight * h
        return CGRect(x: centreX * size.width - w / 2, y: baseY * size.height - h, width: w, height: h)
    }
}

/// The s04 crop geometry in stump heights from the stump centre-x and base-y
/// (stump_crops.py with --inner 0.1; also in the .mlpackage metadata).
enum StumpCrops {
    static let outer: CGFloat = 2.3   // away from the stumps
    static let inner: CGFloat = 0.1   // past the stump centre
    static let above: CGFloat = 3.1   // above the stump base
    static let below: CGFloat = 0.9   // below the stump base
    /// Model input per crop.
    static let inputWidth = 160
    static let inputHeight = 256

    /// (left, right) crops in pixels of a frame of `size`, origin top-left. They may reach
    /// past the frame; the model was trained with black there.
    static func rects(for box: StumpBox, in size: CGSize) -> (left: CGRect, right: CGRect) {
        let h = box.height * size.height
        let cx = box.centreX * size.width
        let top = box.baseY * size.height - above * h
        let w = (outer + inner) * h, ch = (above + below) * h
        return (CGRect(x: cx - outer * h, y: top, width: w, height: ch),
                CGRect(x: cx - inner * h, y: top, width: w, height: ch))
    }
}

/// Full camera frame with the stump box to drag (move) and pinch (resize) onto the
/// bowler's-end stumps, with a yellow line along the stump base across the frame.
struct StumpCalibrationView: View {
    let frame: UIImage?
    @Binding var box: StumpBox
    var onDone: () -> Void

    @State private var dragStart: StumpBox?
    @State private var pinchStart: StumpBox?

    init(frame: UIImage?, box: Binding<StumpBox>, onDone: @escaping () -> Void) {
        self.frame = frame
        self._box = box
        self.onDone = onDone
    }

    var body: some View {
        VStack(spacing: 12) {
            Text("Place the yellow box on the bowler's-end stumps")
                .font(.headline)
                .multilineTextAlignment(.center)
            Text("Drag to move, pinch to resize. Fix the camera first and calibrate again if it moves.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            GeometryReader { geo in
                let size = fitted(in: geo.size, aspect: frameAspect)
                let stumps = box.rect(in: size)
                ZStack(alignment: .topLeading) {
                    if let frame {
                        Image(uiImage: frame).resizable()
                    } else {
                        Color.black.overlay(ProgressView().tint(.white))
                    }
                    // Base line across the frame: lines up with the crease to check tilt.
                    Path { p in
                        p.move(to: CGPoint(x: 0, y: stumps.maxY))
                        p.addLine(to: CGPoint(x: size.width, y: stumps.maxY))
                    }
                    .stroke(.yellow.opacity(0.8), lineWidth: 1.5)
                    Path(stumps).stroke(.yellow, lineWidth: 2.5)
                }
                .frame(width: size.width, height: size.height)
                .clipped()
                .contentShape(Rectangle())
                .gesture(drag(in: size).simultaneously(with: pinch))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            HStack(spacing: 12) {
                Button("Reset") { box = .typical }
                    .buttonStyle(.bordered)
                Button("Done") {
                    box.save()
                    onDone()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
    }

    /// Width/height of the camera frame (portrait 9:16 until the first frame arrives).
    private var frameAspect: CGFloat {
        guard let frame, frame.size.height > 0 else { return 9.0 / 16.0 }
        return frame.size.width / frame.size.height
    }

    private func fitted(in space: CGSize, aspect: CGFloat) -> CGSize {
        guard space.width > 0, space.height > 0 else { return .zero }
        return space.width / space.height > aspect
            ? CGSize(width: space.height * aspect, height: space.height)
            : CGSize(width: space.width, height: space.width / aspect)
    }

    private func drag(in size: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let start = dragStart ?? box
                dragStart = start
                box.centreX = min(max(start.centreX + value.translation.width / size.width, 0), 1)
                box.baseY = min(max(start.baseY + value.translation.height / size.height, 0), 1)
            }
            .onEnded { _ in dragStart = nil }
    }

    private var pinch: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                let start = pinchStart ?? box
                pinchStart = start
                box.height = min(max(start.height * value, 0.04), 0.6)
            }
            .onEnded { _ in pinchStart = nil }
    }
}

/// Drawn over the live full-frame preview: the stump box, a horizontal line along the
/// stump base across the frame and a vertical line from the stump centre to the top.
struct StumpGuideOverlay: View {
    let box: StumpBox

    var body: some View {
        GeometryReader { geo in
            let stumps = box.rect(in: geo.size)
            Path { p in
                p.move(to: CGPoint(x: 0, y: stumps.maxY))
                p.addLine(to: CGPoint(x: geo.size.width, y: stumps.maxY))
                p.move(to: CGPoint(x: stumps.midX, y: stumps.maxY))
                p.addLine(to: CGPoint(x: stumps.midX, y: 0))
            }
            .stroke(.yellow.opacity(0.8), lineWidth: 1)
            Path(stumps).stroke(.yellow, lineWidth: 1.5)
        }
        .allowsHitTesting(false)
    }
}
