"""
Ball Tracker Module
Handles ball detection from model output and Kalman filter-based smoothing.
"""
from __future__ import annotations

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None

from config import Config


class BallTracker:
    """Detects and tracks the cricket ball across video frames using a Kalman filter."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.left_bound, self.right_bound = Config.get_detection_bounds(frame_width)

        # Raw and filtered trajectory storage
        self.raw_positions: list[tuple[int, int]] = []
        self.filtered_positions: list[tuple[int, int]] = []
        self.confidences: list[float] = []
        self.frame_indices: list[int] = []

        # Kalman filter for smooth tracking
        self.kf = self._init_kalman() if KalmanFilter else None
        self.missed_frames = 0
        self.current_frame_idx = 0
        self.tracking_active = False

    # ── Kalman Filter Setup ──────────────────────────────────────────────

    def _init_kalman(self):
        """
        Initialize a 2D Kalman filter.
        State: [x, y, vx, vy]  — position and velocity
        Measurement: [x, y]
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity model)
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # Measurement matrix (we observe x, y)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Covariance matrices
        kf.P *= 1000.0
        kf.R = np.eye(2) * Config.KALMAN_MEASUREMENT_NOISE
        kf.Q = np.eye(4) * Config.KALMAN_PROCESS_NOISE

        return kf

    def reset_kalman(self):
        """Reset the Kalman filter state."""
        self.kf = self._init_kalman() if KalmanFilter else None
        self.tracking_active = False
        self.missed_frames = 0

    # ── Detection Extraction ─────────────────────────────────────────────

    def extract_ball_position(self, detections) -> tuple | None:
        """
        Extract the ball position from supervision Detections object.
        Returns (x_center, y_center, confidence) or None if not detected in the valid zone.
        """
        best_conf = 0.0
        best_pos = None

        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i]
            class_id = detections.class_id[i] if detections.class_id is not None else -1
            confidence = detections.confidence[i] if detections.confidence is not None else 1.0

            if class_id != Config.BALL_CLASS_ID:
                continue

            if confidence < Config.CONFIDENCE_THRESHOLD:
                continue

            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # Check if ball is within the pitch detection strip
            if not (self.left_bound <= x_center <= self.right_bound):
                continue

            if confidence > best_conf:
                best_conf = confidence
                best_pos = (x_center, y_center, confidence)

        return best_pos

    # ── Frame Update ─────────────────────────────────────────────────────

    def update(self, detections, frame_idx: int) -> tuple | None:
        """
        Process a frame's detections. Returns the (filtered_x, filtered_y) or None.
        """
        self.current_frame_idx = frame_idx
        ball = self.extract_ball_position(detections)

        if ball is not None:
            x, y, conf = ball
            self.raw_positions.append((x, y))
            self.confidences.append(conf)
            self.frame_indices.append(frame_idx)
            self.missed_frames = 0

            if self.kf is not None:
                if not self.tracking_active:
                    # Initialize Kalman state with first detection
                    self.kf.x = np.array([x, y, 0, 0], dtype=float)
                    self.tracking_active = True

                self.kf.predict()
                self.kf.update(np.array([x, y], dtype=float))
                fx, fy = int(self.kf.x[0]), int(self.kf.x[1])
            else:
                fx, fy = x, y

            self.filtered_positions.append((fx, fy))
            return (fx, fy)

        else:
            self.missed_frames += 1

            if self.missed_frames > Config.MAX_MISSED_FRAMES:
                # Lost tracking — do not extrapolate
                return None

            # Use Kalman prediction for missed frames
            if self.kf is not None and self.tracking_active:
                self.kf.predict()
                fx, fy = int(self.kf.x[0]), int(self.kf.x[1])

                # Bounds check
                if 0 <= fx < self.frame_width and 0 <= fy < self.frame_height:
                    self.filtered_positions.append((fx, fy))
                    self.frame_indices.append(frame_idx)
                    return (fx, fy)

            return None

    # ── Accessors ────────────────────────────────────────────────────────

    def get_trajectory(self) -> list[tuple[int, int]]:
        """Return the Kalman-filtered trajectory."""
        return self.filtered_positions.copy()

    def get_raw_trajectory(self) -> list[tuple[int, int]]:
        """Return the raw (unfiltered) detections."""
        return self.raw_positions.copy()

    def get_velocity(self) -> tuple[float, float] | None:
        """Return current estimated velocity (vx, vy) from Kalman state."""
        if self.kf is not None and self.tracking_active:
            return (float(self.kf.x[2]), float(self.kf.x[3]))
        if len(self.filtered_positions) >= 2:
            dx = self.filtered_positions[-1][0] - self.filtered_positions[-2][0]
            dy = self.filtered_positions[-1][1] - self.filtered_positions[-2][1]
            return (float(dx), float(dy))
        return None

    def has_enough_data(self) -> bool:
        """Check if we have enough tracked points for prediction."""
        return len(self.filtered_positions) >= Config.MIN_POINTS_FOR_PREDICTION
