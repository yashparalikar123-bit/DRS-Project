"""
DRS Engine Module
Main orchestrator that ties together all DRS components:
ball tracking, trajectory prediction, LBW analysis, pitch map, edge detection, and visualization.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
from inference import get_model

from config import Config
from ball_tracker import BallTracker
from trajectory_predictor import TrajectoryPredictor
from lbw_analyzer import LBWAnalyzer, LBWDecision, Verdict
from pitch_map import PitchMap
from drs_visualizer import DRSVisualizer


class DRSEngine:
    """Main DRS processing engine."""

    def __init__(
        self,
        input_path: str = None,
        output_path: str = None,
        model_id: str = None,
        show_preview: bool = False,
    ):
        self.input_path = input_path or Config.INPUT_VIDEO
        self.output_path = output_path or Config.OUTPUT_VIDEO
        self.model_id = model_id or Config.MODEL_ID
        self.show_preview = show_preview

        # Video properties (set after opening video)
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        self.total_frames = 0

        # Components (initialized after video properties are known)
        self.model = None
        self.tracker: BallTracker | None = None
        self.predictor: TrajectoryPredictor | None = None
        self.lbw_analyzer: LBWAnalyzer | None = None
        self.pitch_map: PitchMap | None = None
        self.visualizer: DRSVisualizer | None = None

        # Supervision annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # State
        self.frames_processed = 0
        self.decision: LBWDecision | None = None

    # ── Initialization ───────────────────────────────────────────────────

    def initialize(self):
        """Load model and initialize all components."""
        print("[DRS] Loading model...")
        self.model = get_model(model_id=self.model_id)
        print(f"[DRS] Model loaded: {self.model_id}")

    def _init_components(self, width: int, height: int):
        """Initialize all DRS sub-components with video dimensions."""
        self.frame_width = width
        self.frame_height = height

        self.tracker = BallTracker(width, height)
        self.predictor = TrajectoryPredictor()
        self.lbw_analyzer = LBWAnalyzer(width, height)
        self.pitch_map = PitchMap(width, height)
        self.visualizer = DRSVisualizer(width, height)

        print(f"[DRS] Components initialized — Frame: {width}x{height}")

        # Print stump positions for debugging
        bt = Config.get_batting_stump_rect()
        bw = Config.get_bowling_stump_rect()
        print(f"[DRS] Batting Stumps: x1={bt['x1']} y1={bt['y1']} x2={bt['x2']} y2={bt['y2']} center_x={bt['center_x']}")
        print(f"[DRS] Bowling Stumps: x1={bw['x1']} y1={bw['y1']} x2={bw['x2']} y2={bw['y2']} center_x={bw['center_x']}")
        print(f"[DRS] Off-stump line at x={Config.BATTING_STUMP_X + Config.OFF_STUMP_OFFSET}")
        print(f"[DRS] Leg-stump line at x={Config.BATTING_STUMP_X - Config.LEG_STUMP_OFFSET}")
        print(f"[DRS] Detection bounds: x=[{Config.DETECTION_LEFT_X}, {Config.DETECTION_RIGHT_X}]")

    # ── Main Processing Pipeline ─────────────────────────────────────────

    def process(self):
        """Run the full DRS pipeline on the input video."""
        self.initialize()

        # Open video
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"[DRS] ERROR: Could not open video: {self.input_path}")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[DRS] Video: {self.input_path}")
        print(f"[DRS] Resolution: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        print(f"[DRS] Total frames: {self.total_frames}")

        # Initialize components
        self._init_components(self.frame_width, self.frame_height)

        # Setup output video
        output_fps = max(1, self.fps * Config.OUTPUT_FPS_FACTOR)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_path, fourcc, output_fps,
            (self.frame_width, self.frame_height),
        )

        print(f"[DRS] Output: {self.output_path} @ {output_fps:.1f} FPS")
        print("[DRS] Processing frames...")

        last_frame = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame.copy()
            annotated = self._process_frame(frame, frame_idx)
            out.write(annotated)

            # Optional live preview
            if self.show_preview:
                preview = cv2.resize(annotated, (0, 0), fx=0.7, fy=0.7)
                cv2.imshow("DRS Review", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[DRS] Preview stopped by user")
                    break

            frame_idx += 1

            # Progress indicator
            if frame_idx % 30 == 0:
                pct = (frame_idx / self.total_frames * 100) if self.total_frames > 0 else 0
                print(f"[DRS] Progress: {frame_idx}/{self.total_frames} ({pct:.0f}%)")

        self.frames_processed = frame_idx
        print(f"[DRS] Processed {frame_idx} frames")

        # ── Post-Processing: Prediction & Verdict ────────────────────
        if last_frame is not None:
            self._generate_verdict(last_frame, out)

        # Cleanup
        cap.release()
        out.release()
        if self.show_preview:
            cv2.destroyAllWindows()

        print(f"[DRS] Output saved: {self.output_path}")
        self._print_summary()

    # ── Per-Frame Processing ─────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process a single video frame through the DRS pipeline."""
        # Run model inference with low confidence to capture all ball detections
        results = self.model.infer(frame, confidence=Config.CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_inference(results)

        # Annotate with supervision
        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections)

        # Track the ball
        ball_pos = self.tracker.update(detections, frame_idx)

        # Draw DRS overlays
        annotated = self.visualizer.draw_drs_label(annotated)

        # Draw ball marker
        if ball_pos:
            annotated = self.visualizer.draw_ball(annotated, ball_pos, detected=True)

        # Draw stump zone
        annotated = self.visualizer.draw_stump_zone(annotated, highlight="neutral")

        # Draw HUD
        trajectory = self.tracker.get_trajectory()
        annotated = self.visualizer.draw_hud(
            annotated,
            frame_idx=frame_idx,
            ball_detected=ball_pos is not None,
            num_tracked=len(trajectory),
            ball_pos=ball_pos,
        )

        return annotated

    # ── Post-Processing & Verdict ────────────────────────────────────────

    def _generate_verdict(self, last_frame: np.ndarray, out: cv2.VideoWriter):
        """Generate trajectory prediction, LBW analysis, and render verdict frames."""
        trajectory = self.tracker.get_trajectory()

        if not self.tracker.has_enough_data():
            print("[DRS] Not enough tracking data for prediction")
            self._write_insufficient_data_frames(last_frame, out)
            return

        # ── Fit trajectory and predict future path ───────────────────
        self.predictor.fit(trajectory)
        predicted_path = self.predictor.predict_future(trajectory)
        smooth_trajectory = self.predictor.get_smooth_trajectory(trajectory)

        print(f"[DRS] Trajectory: {len(trajectory)} points → {len(predicted_path)} predicted")

        # ── Run LBW Analysis ─────────────────────────────────────────
        self.decision = self.lbw_analyzer.analyze(
            trajectory=trajectory,
            predicted_path=predicted_path,
        )
        print(f"\n[DRS] LBW Analysis:\n{self.decision.summary()}\n")

        hitting = self.decision.wicket.value == "Hitting Stumps"
        stump_highlight = "hit" if hitting else "miss"

        # ── Hold frame with stumps highlighted ──────────────────────
        hold_frame = last_frame.copy()
        hold_frame = self.visualizer.draw_stump_zone(hold_frame, highlight=stump_highlight)
        hold_frame = self.visualizer.draw_drs_label(hold_frame)

        # Overlay pitch map
        pitch_map_img = self.pitch_map.render(
            pitching_point=self.decision.pitching_point,
            impact_point=self.decision.impact_point,
            wicket_point=self.decision.wicket_point,
            pitching_in_line=self.decision.pitching.value.endswith("In Line"),
            impact_in_line=self.decision.impact.value.endswith("In Line"),
            hitting_stumps=hitting,
        )
        hold_frame = self.pitch_map.overlay_on_frame(hold_frame, pitch_map_img)

        fps_out = max(1, self.fps * Config.OUTPUT_FPS_FACTOR)
        # Brief hold — just 2 frames
        out.write(hold_frame)
        out.write(hold_frame)

        # ── Verdict screen — a few frames ────────────────────────────
        print("[DRS] Rendering verdict...")
        verdict_frame = self.visualizer.create_verdict_frame(hold_frame, self.decision)
        verdict_count = max(2, int(fps_out * 1))  # ~1 second
        for _ in range(verdict_count):
            out.write(verdict_frame)

    def _write_insufficient_data_frames(
        self, last_frame: np.ndarray, out: cv2.VideoWriter
    ):
        """Write frames indicating insufficient data."""
        frame = last_frame.copy()
        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)

        text = "INSUFFICIENT DATA"
        font = cv2.FONT_HERSHEY_SIMPLEX
        ts = cv2.getTextSize(text, font, 1.2, 3)[0]
        x = (self.frame_width - ts[0]) // 2
        y = self.frame_height // 2
        cv2.putText(frame, text, (x, y), font, 1.2, (0, 200, 255), 3, cv2.LINE_AA)

        sub_text = "Not enough ball detections for DRS review"
        ts2 = cv2.getTextSize(sub_text, font, 0.6, 1)[0]
        cv2.putText(frame, sub_text,
                     ((self.frame_width - ts2[0]) // 2, y + 40),
                     font, 0.6, Config.COLOR_WHITE, 1, cv2.LINE_AA)

        fps_out = max(1, self.fps * Config.OUTPUT_FPS_FACTOR)
        for _ in range(int(fps_out * 3)):
            out.write(frame)

    # ── Summary ──────────────────────────────────────────────────────────

    def _print_summary(self):
        """Print final DRS summary to console."""
        print("\n" + "=" * 60)
        print("  DRS REVIEW COMPLETE")
        print("=" * 60)
        print(f"  Frames processed:  {self.frames_processed}")
        print(f"  Ball detections:   {len(self.tracker.get_raw_trajectory())}")
        print(f"  Tracked points:    {len(self.tracker.get_trajectory())}")
        print()
        if self.decision:
            print(self.decision.summary())
        else:
            print("  No LBW decision generated")
        print("=" * 60 + "\n")
