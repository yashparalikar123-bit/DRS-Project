
from __future__ import annotations

import cv2
import numpy as np

from config import Config


class EdgeDetector:
    """Detects edges/snicks using frame difference analysis near the ball."""

    def __init__(self):
        self.prev_gray: np.ndarray | None = None
        self.edge_events: list[dict] = []

    def check_edge(
        self,
        frame: np.ndarray,
        ball_position: tuple[int, int] | None,
        frame_idx: int,
    ) -> bool:
        """
        Check for an edge event near the ball position.
        Uses frame differencing to detect sudden pixel changes around the ball.

        Returns True if an edge is detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or ball_position is None:
            self.prev_gray = gray
            return False

        # Define ROI around the ball
        bx, by = ball_position
        pad = Config.EDGE_ROI_PADDING
        h, w = gray.shape

        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + pad)
        y2 = min(h, by + pad)

        # Frame difference in the ROI
        roi_curr = gray[y1:y2, x1:x2]
        roi_prev = self.prev_gray[y1:y2, x1:x2]

        if roi_curr.shape != roi_prev.shape or roi_curr.size == 0:
            self.prev_gray = gray
            return False

        diff = cv2.absdiff(roi_curr, roi_prev)
        _, thresh = cv2.threshold(diff, Config.EDGE_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Count significantly changed pixels
        changed = cv2.countNonZero(thresh)

        self.prev_gray = gray

        if changed > Config.EDGE_SPIKE_THRESHOLD:
            self.edge_events.append({
                "frame_idx": frame_idx,
                "position": ball_position,
                "intensity": changed,
            })
            return True

        return False

    def get_edge_events(self) -> list[dict]:
        """Return all detected edge events."""
        return self.edge_events.copy()

    def had_edge(self) -> bool:
        """Check if any edge was detected during the delivery."""
        return len(self.edge_events) > 0

    def get_strongest_edge(self) -> dict | None:
        """Return the edge event with the highest intensity."""
        if not self.edge_events:
            return None
        return max(self.edge_events, key=lambda e: e["intensity"])

    def reset(self):
        """Reset edge detection state for a new delivery."""
        self.prev_gray = None
        self.edge_events.clear()
