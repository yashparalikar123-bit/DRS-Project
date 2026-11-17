"""
DRS System Configuration
All configurable parameters for the Decision Review System.

This is designed for a 2D broadcast cricket view.
Stumps are visible at fixed pixel positions in the frame.
Adjust the STUMP pixel coordinates below to match your video.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the DRS system."""

    # ── API & Model ──────────────────────────────────────────────────────
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
    MODEL_ID = os.getenv("MODEL_ID", "drs-5gvgi/3")

    # ── Video I/O ────────────────────────────────────────────────────────
    INPUT_VIDEO = os.getenv("INPUT_VIDEO", "v16_crop.mp4")
    OUTPUT_VIDEO = os.getenv("OUTPUT_VIDEO", "yash.mp4")
    OUTPUT_FPS_FACTOR = 0.3  # Slow-motion factor for output

    # ── Detection ────────────────────────────────────────────────────────
    BALL_CLASS_ID = 0  # Class ID for the cricket ball in the model
    CONFIDENCE_THRESHOLD = 0.05  # Minimum confidence for ball detection (model outputs low conf)

    # ── Kalman Filter ────────────────────────────────────────────────────
    KALMAN_PROCESS_NOISE = 1e-2
    KALMAN_MEASUREMENT_NOISE = 1e-1
    MAX_MISSED_FRAMES = 8  # Max consecutive missed detections before reset

    # ── Trajectory Prediction ────────────────────────────────────────────
    POLY_DEGREE = 2  # Polynomial degree for curve fitting
    MIN_POINTS_FOR_PREDICTION = 5  # Minimum tracked points before predicting
    NUM_FUTURE_FRAMES = 30  # Number of predicted future points
    PREDICTION_STEP = 1.0  # Step size for future point generation

    # ══════════════════════════════════════════════════════════════════════
    #  2D BROADCAST VIEW — STUMP & PITCH COORDINATES (PIXELS)
    #
    #  These are PIXEL positions in the video frame.
    #  Open the video, find the stumps, and set these values.
    #  The debug overlay will show these positions with colors
    #  so you can fine-tune them.
    #
    #  Video resolution: 1616 x 720
    # ══════════════════════════════════════════════════════════════════════

    # ── Batting End Stumps (near batsman — this is the target for LBW) ───
    # The batsman's end is FARTHER from camera = lower Y values (higher on screen)
    BATTING_STUMP_X = 806       # Center x of the batting-end stumps
    BATTING_STUMP_TOP_Y = 180   # Top of stumps (bail level)
    BATTING_STUMP_BOTTOM_Y = 300  # Bottom of stumps (ground)
    BATTING_STUMP_WIDTH = 35    # Appears smaller due to perspective

    # ── Bowling End Stumps (near bowler — ball is released from here) ────
    # The bowler's end is CLOSER to camera = higher Y values (lower on screen)
    BOWLING_STUMP_X = 810       # Center x of the bowling-end stumps
    BOWLING_STUMP_TOP_Y = 450   # Top of stumps (bail level)
    BOWLING_STUMP_BOTTOM_Y = 650  # Bottom of stumps (ground)
    BOWLING_STUMP_WIDTH = 50    # Appears bigger, closer to camera

    # ── Pitch Strip (the area between the two sets of stumps) ────────────
    # Connects batting end (top, smaller) to bowling end (bottom, bigger)
    PITCH_STRIP_LEFT_TOP = 789      # Left edge at batting end (top)
    PITCH_STRIP_RIGHT_TOP = 824     # Right edge at batting end (top)
    PITCH_STRIP_LEFT_BOTTOM = 785   # Left edge at bowling end (bottom)
    PITCH_STRIP_RIGHT_BOTTOM = 835  # Right edge at bowling end (bottom)

    # ── Detection Bounds (only track ball within this horizontal range) ──
    DETECTION_LEFT_X = 600   # Don't track ball left of this pixel
    DETECTION_RIGHT_X = 1016  # Don't track ball right of this pixel

    # ── LBW Off/Leg Stump Lines ──────────────────────────────────────────
    # These extend the stump width for "in line" judgment
    # Measured from the center of batting stumps
    OFF_STUMP_OFFSET = 20   # Pixels right of center = off stump line
    LEG_STUMP_OFFSET = 20   # Pixels left of center = leg stump line

    # ── Edge Detection ───────────────────────────────────────────────────
    EDGE_DIFF_THRESHOLD = 30  # Pixel intensity change threshold
    EDGE_ROI_PADDING = 50  # Pixels around ball to check for edge
    EDGE_SPIKE_THRESHOLD = 500  # Minimum changed pixels to flag edge

    # ── Visualization Colors (BGR) ───────────────────────────────────────
    COLOR_BALL_DETECTED = (0, 255, 0)      # Green
    COLOR_TRAJECTORY = (0, 100, 255)       # Orange
    COLOR_TRAJECTORY_DOTS = (0, 0, 255)    # Red
    COLOR_PREDICTED = (0, 255, 255)        # Yellow
    COLOR_STUMP_HIT = (0, 0, 255)          # Red — hitting stumps
    COLOR_STUMP_MISS = (0, 255, 0)         # Green — missing stumps
    COLOR_PITCH_INLINE = (0, 200, 0)       # Green — pitched in line
    COLOR_PITCH_OUTSIDE = (0, 0, 255)      # Red — pitched outside
    COLOR_IMPACT_INLINE = (0, 200, 0)      # Green — impact in line
    COLOR_IMPACT_OUTSIDE = (0, 0, 255)     # Red — impact outside
    COLOR_OUT_TEXT = (0, 0, 255)            # Red
    COLOR_NOT_OUT_TEXT = (0, 255, 0)        # Green
    COLOR_OVERLAY_BG = (0, 0, 0)           # Black background for overlays
    COLOR_WHITE = (255, 255, 255)

    # Debug colors for stumps (bright and distinct)
    COLOR_BATTING_STUMPS = (255, 0, 255)   # Magenta — batting end
    COLOR_BOWLING_STUMPS = (255, 255, 0)   # Cyan — bowling end
    COLOR_PITCH_STRIP = (200, 150, 50)     # Light blue — pitch strip
    COLOR_STUMP_LINE = (0, 165, 255)       # Orange — off/leg stump guide lines
    COLOR_DEBUG_TEXT = (0, 255, 255)        # Yellow — debug coordinate text

    # ── Visualization Sizes ──────────────────────────────────────────────
    TRAJECTORY_THICKNESS = 3
    PREDICTED_THICKNESS = 2
    BALL_DOT_RADIUS = 6
    STUMP_LINE_THICKNESS = 2
    FONT = None  # Will use cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    OVERLAY_ALPHA = 0.6

    # ── DRS Replay ───────────────────────────────────────────────────────
    REPLAY_HOLD_SECONDS = 1  # Seconds to hold final DRS frame
    VERDICT_DISPLAY_SECONDS = 2  # Seconds to show verdict screen
    SLOWMO_FACTOR = 0.25  # Speed factor for replay section

    # ── Debug Mode ───────────────────────────────────────────────────────
    DEBUG_SHOW_COORDS = True  # Show pixel coordinates on stumps
    DEBUG_SHOW_STUMP_LABELS = True  # Label each stump end

    @classmethod
    def get_batting_stump_rect(cls) -> dict:
        """Get batting-end stump rectangle in pixel coordinates."""
        half_w = cls.BATTING_STUMP_WIDTH // 2
        return {
            "x1": cls.BATTING_STUMP_X - half_w,
            "y1": cls.BATTING_STUMP_TOP_Y,
            "x2": cls.BATTING_STUMP_X + half_w,
            "y2": cls.BATTING_STUMP_BOTTOM_Y,
            "center_x": cls.BATTING_STUMP_X,
            "top_y": cls.BATTING_STUMP_TOP_Y,
            "bottom_y": cls.BATTING_STUMP_BOTTOM_Y,
        }

    @classmethod
    def get_bowling_stump_rect(cls) -> dict:
        """Get bowling-end stump rectangle in pixel coordinates."""
        half_w = cls.BOWLING_STUMP_WIDTH // 2
        return {
            "x1": cls.BOWLING_STUMP_X - half_w,
            "y1": cls.BOWLING_STUMP_TOP_Y,
            "x2": cls.BOWLING_STUMP_X + half_w,
            "y2": cls.BOWLING_STUMP_BOTTOM_Y,
            "center_x": cls.BOWLING_STUMP_X,
            "top_y": cls.BOWLING_STUMP_TOP_Y,
            "bottom_y": cls.BOWLING_STUMP_BOTTOM_Y,
        }

    @classmethod
    def get_stump_rect(cls, frame_width: int = 0, frame_height: int = 0) -> dict:
        """Get the batting-end stump rect (this is the LBW target). 
        frame_width/frame_height kept for backward compat but not used."""
        return cls.get_batting_stump_rect()

    @classmethod
    def get_detection_bounds(cls, frame_width: int = 0) -> tuple:
        """Return left and right pixel bounds for ball detection."""
        return (cls.DETECTION_LEFT_X, cls.DETECTION_RIGHT_X)
