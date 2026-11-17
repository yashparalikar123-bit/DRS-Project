"""
Trajectory Predictor Module
Uses polynomial curve fitting to predict the ball's future path.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

from config import Config


class TrajectoryPredictor:
    """Predicts future ball trajectory using polynomial / spline fitting."""

    def __init__(self):
        self.poly_x = None  # Polynomial for x(t)
        self.poly_y = None  # Polynomial for y(t)
        self.trajectory_fitted = False

    def fit(self, positions: list[tuple[int, int]]) -> bool:
        """
        Fit polynomials to the trajectory.
        Uses parameter t (0..N-1) and fits x(t) and y(t) separately.
        Returns True if fitting succeeded.
        """
        if len(positions) < Config.MIN_POINTS_FOR_PREDICTION:
            return False

        n = len(positions)
        t = np.arange(n, dtype=float)
        xs = np.array([p[0] for p in positions], dtype=float)
        ys = np.array([p[1] for p in positions], dtype=float)

        try:
            degree = min(Config.POLY_DEGREE, n - 1)
            self.poly_x = np.polyfit(t, xs, degree)
            self.poly_y = np.polyfit(t, ys, degree)
            self.trajectory_fitted = True
            return True
        except (np.linalg.LinAlgError, ValueError):
            self.trajectory_fitted = False
            return False

    def predict_future(
        self,
        positions: list[tuple[int, int]],
        num_future: int = None,
        step: float = None,
    ) -> list[tuple[int, int]]:
        """
        Predict future ball positions beyond the last observed point.
        Direction-agnostic: follows the ball's natural direction of travel.
        Uses the "active" portion of the trajectory (where the ball was
        actually moving significantly) to compute velocity.
        """
        if len(positions) < 2:
            return []

        num_future = num_future or Config.NUM_FUTURE_FRAMES

        # ── Find the active portion of the trajectory ────────────────
        # The Kalman filter may produce stale predictions at the end.
        # Find the segment with the most movement by looking at where
        # the ball was actively travelling (not stale/repeated points).
        # Use the unique-movement portion: trim repeated points at the end.
        active = list(positions)
        while len(active) > 5 and active[-1] == active[-2]:
            active.pop()

        # Compute velocity from the active segment
        if len(active) >= 2:
            # Use the last ~40% of active trajectory for velocity
            tail_n = max(5, len(active) * 2 // 5)
            tail = active[-tail_n:]
            dx = tail[-1][0] - tail[0][0]
            dy = tail[-1][1] - tail[0][1]
            n_tail = len(tail)
            vx = dx / n_tail
            vy = dy / n_tail
        else:
            vx = 0.0
            vy = 1.0

        # If velocity is negligible, use full trajectory average
        if abs(vx) < 0.1 and abs(vy) < 0.1 and len(active) >= 2:
            dx = active[-1][0] - active[0][0]
            dy = active[-1][1] - active[0][1]
            n_full = len(active)
            vx = dx / n_full
            vy = dy / n_full

        # Start extrapolation from the last active point
        start_x, start_y = active[-1]

        # Generate prediction points following the direction of travel
        future_points = []
        max_steps = max(num_future, 500)
        bat_top = Config.BATTING_STUMP_TOP_Y
        bat_bot = Config.BATTING_STUMP_BOTTOM_Y

        for i in range(1, max_steps + 1):
            fx = int(start_x + i * vx)
            fy = int(start_y + i * vy)
            future_points.append((fx, fy))
            # Stop once we've gone well past the batting stump zone
            # Works regardless of whether ball travels up or down
            if vy > 0 and fy > bat_bot + 100:
                break
            if vy < 0 and fy < bat_top - 100:
                break
            if fy < 0 or fy > 1500:
                break

        return future_points

    def predict_at_y(
        self, positions: list[tuple[int, int]], target_y: int
    ) -> tuple[int, int] | None:
        """
        Predict the x-coordinate when the ball reaches a specific y-coordinate.
        Direction-agnostic: works whether ball travels up or down.
        """
        if len(positions) < 2:
            return None

        # Trim stale tail
        active = list(positions)
        while len(active) > 5 and active[-1] == active[-2]:
            active.pop()

        if len(active) < 2:
            return None

        # Use tail velocity
        tail_n = max(5, len(active) * 2 // 5)
        tail = active[-tail_n:]
        dx = tail[-1][0] - tail[0][0]
        dy = tail[-1][1] - tail[0][1]
        n_tail = len(tail)
        vx = dx / n_tail
        vy = dy / n_tail

        if abs(vy) < 0.01:
            return None

        start_x, start_y = active[-1]
        steps = (target_y - start_y) / vy
        # Steps should be positive (in the future)
        if steps < 0:
            return None

        fx = int(start_x + steps * vx)
        return (fx, target_y)

    def get_smooth_trajectory(
        self, positions: list[tuple[int, int]], num_points: int = 100
    ) -> list[tuple[int, int]]:
        """
        Return a smoothed version of the observed trajectory using spline interpolation.
        """
        if len(positions) < 4:
            return positions

        n = len(positions)
        t = np.arange(n, dtype=float)
        xs = np.array([p[0] for p in positions], dtype=float)
        ys = np.array([p[1] for p in positions], dtype=float)

        try:
            t_smooth = np.linspace(0, n - 1, num_points)
            spline_x = UnivariateSpline(t, xs, k=min(3, n - 1), s=n * 2)
            spline_y = UnivariateSpline(t, ys, k=min(3, n - 1), s=n * 2)
            smooth_xs = spline_x(t_smooth)
            smooth_ys = spline_y(t_smooth)
            return [(int(x), int(y)) for x, y in zip(smooth_xs, smooth_ys)]
        except Exception:
            return positions

    def _fallback_prediction(
        self, positions: list[tuple[int, int]], num_future: int
    ) -> list[tuple[int, int]]:
        """Linear fallback when polynomial fitting fails."""
        if len(positions) < 2:
            return []

        # Average velocity from last few points
        num_pts = min(5, len(positions) - 1)
        dx_total = positions[-1][0] - positions[-1 - num_pts][0]
        dy_total = positions[-1][1] - positions[-1 - num_pts][1]
        avg_dx = dx_total / num_pts
        avg_dy = dy_total / num_pts

        x_last, y_last = positions[-1]
        future = []
        for i in range(1, num_future + 1):
            fx = int(x_last + i * avg_dx)
            fy = int(y_last + i * avg_dy)
            future.append((fx, fy))
        return future
