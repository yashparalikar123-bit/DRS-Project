"""
Pitch Map Module
Renders a top-down pitch map showing where the ball pitched, impacted,
and whether it was going on to hit the stumps.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import Config


class PitchMap:
    """Renders a top-down cricket pitch map overlay."""

    # Pitch map dimensions (rendered as a small overlay)
    MAP_WIDTH = 160
    MAP_HEIGHT = 400
    MARGIN = 10

    # Stump positions on the map
    STUMP_MAP_Y_TOP = 20
    STUMP_MAP_Y_BOTTOM = 60
    STUMP_MAP_CENTER_X = MAP_WIDTH // 2

    # Pitch zones on the map
    GOOD_LENGTH_TOP = 100
    GOOD_LENGTH_BOTTOM = 200
    FULL_LENGTH_TOP = 200
    FULL_LENGTH_BOTTOM = 300
    CREASE_Y = 340

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def render(
        self,
        pitching_point: tuple[int, int] | None = None,
        impact_point: tuple[int, int] | None = None,
        wicket_point: tuple[int, int] | None = None,
        pitching_in_line: bool = True,
        impact_in_line: bool = True,
        hitting_stumps: bool = False,
    ) -> np.ndarray:
        """
        Create a pitch map image with marked positions.
        Returns an RGBA or BGR image of the pitch map.
        """
        # Create a dark pitch map background
        pitch_map = np.zeros((self.MAP_HEIGHT, self.MAP_WIDTH, 3), dtype=np.uint8)
        pitch_map[:] = (30, 60, 30)  # Dark green

        # Draw the pitch strip (tan/brown color)
        strip_left = self.MAP_WIDTH // 2 - 30
        strip_right = self.MAP_WIDTH // 2 + 30
        cv2.rectangle(
            pitch_map,
            (strip_left, 10),
            (strip_right, self.MAP_HEIGHT - 10),
            (140, 180, 200),  # Light tan
            -1,
        )

        # Draw pitch zone lines
        for y in [self.GOOD_LENGTH_TOP, self.GOOD_LENGTH_BOTTOM,
                   self.FULL_LENGTH_TOP, self.FULL_LENGTH_BOTTOM]:
            cv2.line(pitch_map, (strip_left, y), (strip_right, y),
                     (100, 140, 160), 1)

        # Draw crease line
        cv2.line(
            pitch_map,
            (strip_left - 10, self.CREASE_Y),
            (strip_right + 10, self.CREASE_Y),
            Config.COLOR_WHITE,
            2,
        )

        # Draw stump positions (bowler's end)
        stump_w = 12
        cv2.rectangle(
            pitch_map,
            (self.STUMP_MAP_CENTER_X - stump_w // 2, self.STUMP_MAP_Y_TOP),
            (self.STUMP_MAP_CENTER_X + stump_w // 2, self.STUMP_MAP_Y_BOTTOM),
            Config.COLOR_WHITE,
            1,
        )

        # Draw stump positions (batsman's end) — this is our target
        batsman_stump_y = self.CREASE_Y + 10
        stump_color = Config.COLOR_STUMP_HIT if hitting_stumps else Config.COLOR_STUMP_MISS
        cv2.rectangle(
            pitch_map,
            (self.STUMP_MAP_CENTER_X - stump_w // 2, batsman_stump_y),
            (self.STUMP_MAP_CENTER_X + stump_w // 2, batsman_stump_y + 30),
            stump_color,
            2,
        )

        # Draw off-stump and leg-stump guide lines
        off_x = self.STUMP_MAP_CENTER_X + stump_w // 2
        leg_x = self.STUMP_MAP_CENTER_X - stump_w // 2
        cv2.line(pitch_map, (off_x, 10), (off_x, self.MAP_HEIGHT - 10),
                 (100, 100, 100), 1, cv2.LINE_AA)
        cv2.line(pitch_map, (leg_x, 10), (leg_x, self.MAP_HEIGHT - 10),
                 (100, 100, 100), 1, cv2.LINE_AA)

        # ── Mark pitching point ──────────────────────────────────────
        if pitching_point is not None:
            map_pt = self._frame_to_map(pitching_point)
            color = Config.COLOR_PITCH_INLINE if pitching_in_line else Config.COLOR_PITCH_OUTSIDE
            cv2.circle(pitch_map, map_pt, 8, color, -1)
            cv2.circle(pitch_map, map_pt, 8, Config.COLOR_WHITE, 1)

        # ── Mark impact point ────────────────────────────────────────
        if impact_point is not None:
            map_pt = self._frame_to_map(impact_point)
            color = Config.COLOR_IMPACT_INLINE if impact_in_line else Config.COLOR_IMPACT_OUTSIDE
            cv2.circle(pitch_map, map_pt, 8, color, -1)
            cv2.circle(pitch_map, map_pt, 8, Config.COLOR_WHITE, 1)
            # X marker for impact
            cv2.drawMarker(
                pitch_map, map_pt, Config.COLOR_WHITE,
                cv2.MARKER_CROSS, 12, 2
            )

        # ── Mark wicket intersection ─────────────────────────────────
        if wicket_point is not None:
            map_pt = self._frame_to_map_wicket(wicket_point)
            color = Config.COLOR_STUMP_HIT if hitting_stumps else Config.COLOR_STUMP_MISS
            cv2.circle(pitch_map, map_pt, 6, color, -1)

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pitch_map, "PITCH MAP", (25, self.MAP_HEIGHT - 3),
                     font, 0.4, Config.COLOR_WHITE, 1)

        # Add border
        cv2.rectangle(pitch_map, (0, 0),
                       (self.MAP_WIDTH - 1, self.MAP_HEIGHT - 1),
                       Config.COLOR_WHITE, 1)

        return pitch_map

    def _frame_to_map(self, point: tuple[int, int]) -> tuple[int, int]:
        """Convert a frame coordinate to pitch map coordinate."""
        x_ratio = point[0] / self.frame_width
        y_ratio = point[1] / self.frame_height

        map_x = int(self.MAP_WIDTH * 0.2 + x_ratio * self.MAP_WIDTH * 0.6)
        map_y = int(self.GOOD_LENGTH_TOP + y_ratio * (self.CREASE_Y - self.GOOD_LENGTH_TOP))

        # Clamp to map bounds
        map_x = max(5, min(self.MAP_WIDTH - 5, map_x))
        map_y = max(5, min(self.MAP_HEIGHT - 5, map_y))
        return (map_x, map_y)

    def _frame_to_map_wicket(self, point: tuple[int, int]) -> tuple[int, int]:
        """Convert a wicket intersection point to map coordinate near the batsman stump."""
        x_ratio = point[0] / self.frame_width
        map_x = int(self.MAP_WIDTH * 0.2 + x_ratio * self.MAP_WIDTH * 0.6)
        map_y = self.CREASE_Y + 25
        map_x = max(5, min(self.MAP_WIDTH - 5, map_x))
        return (map_x, map_y)

    def overlay_on_frame(
        self,
        frame: np.ndarray,
        pitch_map: np.ndarray,
        position: str = "top-right",
    ) -> np.ndarray:
        """Overlay the pitch map on a corner of the video frame."""
        fh, fw = frame.shape[:2]
        mh, mw = pitch_map.shape[:2]

        if position == "top-right":
            x_off = fw - mw - self.MARGIN
            y_off = self.MARGIN
        elif position == "top-left":
            x_off = self.MARGIN
            y_off = self.MARGIN
        elif position == "bottom-right":
            x_off = fw - mw - self.MARGIN
            y_off = fh - mh - self.MARGIN
        else:
            x_off = self.MARGIN
            y_off = fh - mh - self.MARGIN

        # Blend the pitch map onto the frame
        roi = frame[y_off: y_off + mh, x_off: x_off + mw]
        blended = cv2.addWeighted(roi, 0.3, pitch_map, 0.7, 0)
        frame[y_off: y_off + mh, x_off: x_off + mw] = blended

        return frame
