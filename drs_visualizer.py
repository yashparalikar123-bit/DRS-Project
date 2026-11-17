"""
DRS Visualizer Module
Renders Hawk-Eye style overlays, stump zones, trajectories, and verdict screens.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import Config
from lbw_analyzer import LBWDecision, Verdict, PitchingResult, ImpactResult, WicketResult


class DRSVisualizer:
    """Renders all DRS visual elements on video frames."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.batting_stumps = Config.get_batting_stump_rect()
        self.bowling_stumps = Config.get_bowling_stump_rect()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Ball & Trajectory Drawing ────────────────────────────────────────

    def draw_ball(
        self, frame: np.ndarray, position: tuple[int, int], detected: bool = True
    ) -> np.ndarray:
        """Draw the ball marker on the frame."""
        color = Config.COLOR_BALL_DETECTED if detected else (150, 150, 150)
        cv2.circle(frame, position, Config.BALL_DOT_RADIUS, color, -1)
        cv2.circle(frame, position, Config.BALL_DOT_RADIUS + 1, Config.COLOR_WHITE, 1)
        return frame

    def draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: list[tuple[int, int]],
        color: tuple[int, int, int] = None,
        thickness: int = None,
        draw_dots: bool = True,
    ) -> np.ndarray:
        """Draw the ball trajectory path on the frame."""
        if len(trajectory) < 2:
            return frame

        color = color or Config.COLOR_TRAJECTORY
        thickness = thickness or Config.TRAJECTORY_THICKNESS

        # Draw smooth line
        pts = np.array(trajectory, dtype=np.int32)
        cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)

        # Draw dots at each position
        if draw_dots:
            for pt in trajectory:
                cv2.circle(frame, pt, 4, Config.COLOR_TRAJECTORY_DOTS, -1)

        return frame

    def draw_predicted_path(
        self,
        frame: np.ndarray,
        predicted: list[tuple[int, int]],
        hitting_stumps: bool = False,
    ) -> np.ndarray:
        """Draw the predicted future path (dashed style)."""
        if len(predicted) < 2:
            return frame

        color = Config.COLOR_STUMP_HIT if hitting_stumps else Config.COLOR_PREDICTED

        # Draw dashed line effect
        for i in range(len(predicted) - 1):
            if i % 2 == 0:  # Skip every other segment for dashed effect
                cv2.line(
                    frame,
                    predicted[i],
                    predicted[i + 1],
                    color,
                    Config.PREDICTED_THICKNESS,
                    cv2.LINE_AA,
                )

        # Draw dots with fading effect
        for i, pt in enumerate(predicted):
            alpha = max(0.3, 1.0 - i / len(predicted))
            radius = max(2, int(Config.BALL_DOT_RADIUS * alpha))
            cv2.circle(frame, pt, radius, color, -1)

        return frame

    # ── Pitch Strip & Stump Drawing (2D Broadcast View) ────────────────

    def draw_pitch_strip(self, frame: np.ndarray) -> np.ndarray:
        """Draw the pitch strip between batting and bowling end stumps."""
        # 4-point polygon for the pitch strip (trapezoid due to perspective)
        # Top = batting end (smaller, higher on screen)
        # Bottom = bowling end (bigger, lower on screen)
        pts = np.array([
            [Config.PITCH_STRIP_LEFT_TOP, Config.BATTING_STUMP_TOP_Y],
            [Config.PITCH_STRIP_RIGHT_TOP, Config.BATTING_STUMP_TOP_Y],
            [Config.PITCH_STRIP_RIGHT_BOTTOM, Config.BOWLING_STUMP_BOTTOM_Y],
            [Config.PITCH_STRIP_LEFT_BOTTOM, Config.BOWLING_STUMP_BOTTOM_Y],
        ], dtype=np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], Config.COLOR_PITCH_STRIP)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Draw pitch strip outline
        cv2.polylines(frame, [pts], True, Config.COLOR_PITCH_STRIP, 1, cv2.LINE_AA)

        return frame

    def draw_stumps(
        self,
        frame: np.ndarray,
        highlight: str = "neutral",  # "hit", "miss", "neutral"
    ) -> np.ndarray:
        """Draw both sets of stumps with debug colors and coordinate labels."""
        # ── Draw pitch strip first ───────────────────────────────────
        frame = self.draw_pitch_strip(frame)

        # ── Bowling End Stumps (Cyan) ────────────────────────────────
        bs = self.bowling_stumps
        frame = self._draw_one_stump_set(
            frame, bs, Config.COLOR_BOWLING_STUMPS, "BOWLING END"
        )

        # ── Batting End Stumps (Magenta) — This is the LBW target ────
        bt = self.batting_stumps
        if highlight == "hit":
            bat_color = Config.COLOR_STUMP_HIT
        elif highlight == "miss":
            bat_color = Config.COLOR_STUMP_MISS
        else:
            bat_color = Config.COLOR_BATTING_STUMPS
        frame = self._draw_one_stump_set(
            frame, bt, bat_color, "BATTING END (LBW TARGET)"
        )

        # ── Off-stump & Leg-stump guide lines REMOVED per user request ──

        return frame

    def _draw_one_stump_set(
        self,
        frame: np.ndarray,
        sr: dict,
        color: tuple,
        label: str,
    ) -> np.ndarray:
        """Draw a single set of 3 stumps with bails."""
        x1, y1, x2, y2 = sr["x1"], sr["y1"], sr["x2"], sr["y2"]
        cx = sr["center_x"]

        # Semi-transparent stump zone fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Stump outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 3 individual stumps
        stump_gap = (x2 - x1) / 4
        for i in range(1, 4):
            sx = int(x1 + i * stump_gap)
            cv2.line(frame, (sx, y1), (sx, y2), color, 2)

        # Bails on top
        cv2.line(frame, (x1 - 4, y1), (x2 + 4, y1), color, 3)

        # ── Debug: coordinate labels ─────────────────────────────────
        if Config.DEBUG_SHOW_COORDS:
            # Top-left coordinate
            cv2.putText(frame, f"({x1},{y1})",
                        (x1 - 50, y1 - 8),
                        self.font, 0.35, Config.COLOR_DEBUG_TEXT, 1, cv2.LINE_AA)
            # Bottom-right coordinate
            cv2.putText(frame, f"({x2},{y2})",
                        (x2 + 5, y2 + 12),
                        self.font, 0.35, Config.COLOR_DEBUG_TEXT, 1, cv2.LINE_AA)
            # Center
            cv2.circle(frame, (cx, (y1 + y2) // 2), 3, Config.COLOR_DEBUG_TEXT, -1)
            cv2.putText(frame, f"cx={cx}",
                        (cx + 8, (y1 + y2) // 2 + 4),
                        self.font, 0.35, Config.COLOR_DEBUG_TEXT, 1, cv2.LINE_AA)

        # Label
        if Config.DEBUG_SHOW_STUMP_LABELS:
            cv2.putText(frame, label,
                        (x1 - 10, y1 - 22),
                        self.font, 0.4, color, 1, cv2.LINE_AA)

        return frame

    # Backward-compatible alias
    def draw_stump_zone(
        self,
        frame: np.ndarray,
        highlight: str = "neutral",
    ) -> np.ndarray:
        """Draw stumps — calls the new draw_stumps method."""
        return self.draw_stumps(frame, highlight)

    # ── Info Overlays ────────────────────────────────────────────────────

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        ball_detected: bool,
        num_tracked: int,
        ball_pos: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Draw heads-up display info on the frame."""
        y_offset = 30
        line_height = 25

        # Semi-transparent background
        num_lines = 4 + (1 if ball_pos else 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 5 + line_height * num_lines), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        info_lines = [
            ("DRS BALL TRACKING", (0, 200, 255)),
            (f"Frame: {frame_idx}", Config.COLOR_WHITE),
            (f"Ball: {'DETECTED' if ball_detected else 'SEARCHING...'}", 
             Config.COLOR_BALL_DETECTED if ball_detected else Config.COLOR_WHITE),
            (f"Tracked Points: {num_tracked}", Config.COLOR_WHITE),
        ]

        if ball_pos:
            info_lines.append((f"Ball XY: ({ball_pos[0]}, {ball_pos[1]})", Config.COLOR_DEBUG_TEXT))

        for i, (line, color) in enumerate(info_lines):
            cv2.putText(
                frame, line,
                (10, y_offset + i * line_height),
                self.font, 0.5, color, 1, cv2.LINE_AA,
            )

        return frame

    def draw_drs_label(self, frame: np.ndarray) -> np.ndarray:
        """Draw 'DRS REVIEW' label at the top center."""
        text = "DRS REVIEW IN PROGRESS"
        text_size = cv2.getTextSize(text, self.font, 0.8, 2)[0]
        x = (self.frame_width - text_size[0]) // 2
        y = 35

        # Background
        cv2.rectangle(
            frame,
            (x - 10, y - 25),
            (x + text_size[0] + 10, y + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, text, (x, y), self.font, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        return frame

    # ── Verdict Screen ───────────────────────────────────────────────────

    def create_verdict_frame(
        self,
        base_frame: np.ndarray,
        decision: LBWDecision,
    ) -> np.ndarray:
        """Create the final DRS verdict overlay frame."""
        frame = base_frame.copy()

        # Dark overlay
        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)

        # ── Main verdict box ─────────────────────────────────────────
        box_w, box_h = 500, 320
        bx = (self.frame_width - box_w) // 2
        by = (self.frame_height - box_h) // 2

        # Box background
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), Config.COLOR_WHITE, 2)

        # Title
        title = "DRS DECISION"
        ts = cv2.getTextSize(title, self.font, 1.0, 2)[0]
        cv2.putText(
            frame, title,
            (bx + (box_w - ts[0]) // 2, by + 40),
            self.font, 1.0, (0, 200, 255), 2, cv2.LINE_AA,
        )

        # Separator line
        cv2.line(frame, (bx + 20, by + 55), (bx + box_w - 20, by + 55),
                 Config.COLOR_WHITE, 1)

        # Decision details
        details = [
            ("PITCHING", decision.pitching.value, self._pitching_color(decision.pitching)),
            ("IMPACT", decision.impact.value, self._impact_color(decision.impact)),
            ("WICKETS", decision.wicket.value, self._wicket_color(decision.wicket)),
        ]

        for i, (label, value, color) in enumerate(details):
            y_pos = by + 90 + i * 50
            cv2.putText(
                frame, f"{label}:",
                (bx + 30, y_pos),
                self.font, 0.6, Config.COLOR_WHITE, 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, value,
                (bx + 180, y_pos),
                self.font, 0.6, color, 2, cv2.LINE_AA,
            )

        # ── Verdict ──────────────────────────────────────────────────
        verdict_text = decision.verdict.value
        if decision.verdict == Verdict.OUT:
            verdict_color = Config.COLOR_OUT_TEXT
        elif decision.verdict == Verdict.NOT_OUT:
            verdict_color = Config.COLOR_NOT_OUT_TEXT
        else:
            verdict_color = (0, 255, 255)  # Yellow for umpire's call

        # Verdict background
        vy = by + box_h - 70
        cv2.rectangle(frame, (bx + 20, vy - 5), (bx + box_w - 20, vy + 45),
                       verdict_color, -1)
        vs = cv2.getTextSize(verdict_text, self.font, 1.2, 3)[0]
        cv2.putText(
            frame, verdict_text,
            (bx + (box_w - vs[0]) // 2, vy + 35),
            self.font, 1.2, Config.COLOR_WHITE if decision.verdict != Verdict.NOT_OUT else (0, 0, 0), 3, cv2.LINE_AA,
        )

        # Edge info
        if decision.reasons:
            reason_text = decision.reasons[0] if decision.reasons else ""
            if len(reason_text) > 50:
                reason_text = reason_text[:47] + "..."
            cv2.putText(
                frame, reason_text,
                (bx + 30, by + box_h + 25),
                self.font, 0.45, Config.COLOR_WHITE, 1, cv2.LINE_AA,
            )

        return frame

    def create_edge_alert_frame(
        self, frame: np.ndarray, edge_position: tuple[int, int]
    ) -> np.ndarray:
        """Draw an edge detection alert on the frame."""
        result = frame.copy()

        # Pulsing circle around edge point
        cv2.circle(result, edge_position, 40, (0, 255, 255), 3)
        cv2.circle(result, edge_position, 50, (0, 255, 255), 1)

        # Alert text
        text = "EDGE DETECTED"
        ts = cv2.getTextSize(text, self.font, 0.8, 2)[0]
        tx = edge_position[0] - ts[0] // 2
        ty = edge_position[1] - 60

        cv2.rectangle(result, (tx - 5, ty - 20), (tx + ts[0] + 5, ty + 5),
                       (0, 0, 0), -1)
        cv2.putText(result, text, (tx, ty), self.font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return result

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pitching_color(p: PitchingResult) -> tuple:
        if p == PitchingResult.IN_LINE:
            return Config.COLOR_PITCH_INLINE
        elif p == PitchingResult.OUTSIDE_OFF:
            return (0, 200, 200)
        return Config.COLOR_PITCH_OUTSIDE

    @staticmethod
    def _impact_color(p: ImpactResult) -> tuple:
        if p == ImpactResult.IN_LINE:
            return Config.COLOR_IMPACT_INLINE
        return Config.COLOR_IMPACT_OUTSIDE

    @staticmethod
    def _wicket_color(w: WicketResult) -> tuple:
        if w == WicketResult.HITTING:
            return Config.COLOR_STUMP_HIT
        return Config.COLOR_STUMP_MISS
