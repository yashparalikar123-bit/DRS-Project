
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from config import Config


class PitchingResult(Enum):
    IN_LINE = "Pitched In Line"
    OUTSIDE_OFF = "Pitched Outside Off"
    OUTSIDE_LEG = "Pitched Outside Leg"
    UNKNOWN = "Pitching Unknown"


class ImpactResult(Enum):
    IN_LINE = "Impact In Line"
    OUTSIDE_OFF = "Impact Outside Off"
    OUTSIDE_LEG = "Impact Outside Leg"
    UNKNOWN = "Impact Unknown"


class WicketResult(Enum):
    HITTING = "Hitting Stumps"
    MISSING_OFF = "Missing Off Side"
    MISSING_LEG = "Missing Leg Side"
    MISSING_OVER = "Going Over Stumps"
    MISSING_UNDER = "Going Under (Unlikely)"
    UNKNOWN = "Wicket Unknown"


class Verdict(Enum):
    OUT = "OUT"
    NOT_OUT = "NOT OUT"
    UMPIRES_CALL = "UMPIRE'S CALL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class LBWDecision:
    """Complete LBW decision breakdown."""
    pitching: PitchingResult = PitchingResult.UNKNOWN
    impact: ImpactResult = ImpactResult.UNKNOWN
    wicket: WicketResult = WicketResult.UNKNOWN
    verdict: Verdict = Verdict.INCONCLUSIVE

    pitching_point: tuple[int, int] | None = None
    impact_point: tuple[int, int] | None = None
    wicket_point: tuple[int, int] | None = None

    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Verdict: {self.verdict.value}",
            f"  Pitching: {self.pitching.value}",
            f"  Impact:   {self.impact.value}",
            f"  Wickets:  {self.wicket.value}",
        ]
        if self.reasons:
            lines.append("  Reasons:")
            for r in self.reasons:
                lines.append(f"    - {r}")
        return "\n".join(lines)


class LBWAnalyzer:
    """Analyzes ball trajectory for LBW decisions.

    Simplified logic:
    - Pitching: If the ball's landing point X is within the pitch strip → In Line
    - Wicket:   If the predicted trajectory's endpoint falls within the
                batting stump rectangle (4 corners) → Hitting Stumps
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.stump_rect = Config.get_stump_rect(frame_width, frame_height)

    def analyze(
        self,
        trajectory: list[tuple[int, int]],
        predicted_path: list[tuple[int, int]],
        impact_point: tuple[int, int] | None = None,
    ) -> LBWDecision:
        """Run full LBW analysis on the ball trajectory."""
        decision = LBWDecision()

        if len(trajectory) < Config.MIN_POINTS_FOR_PREDICTION:
            decision.reasons.append("Insufficient tracking data")
            return decision

        # ── Step 1: Pitching — is the ball landing on the pitch strip? ─
        pitching_pt = self._find_pitching_point(trajectory)
        if pitching_pt:
            decision.pitching_point = pitching_pt
            decision.pitching = self._classify_pitching(pitching_pt)
        else:
            decision.pitching = PitchingResult.UNKNOWN
            decision.reasons.append("Could not determine pitching point")

        # ── Step 2 & 3: Find the point where ball is at stump Y level ─
        # This point is used for BOTH impact and wicket judgment
        wicket_pt = self._find_wicket_point(trajectory, predicted_path)
        if wicket_pt:
            # Impact = same point (where ball is at stump height)
            decision.impact_point = wicket_pt
            decision.impact = self._classify_impact(wicket_pt)
            # Wicket = is that point within the stump rectangle?
            decision.wicket_point = wicket_pt
            decision.wicket = self._classify_wicket_hit(wicket_pt)
        else:
            # Fallback: use last trajectory point for impact
            decision.impact_point = trajectory[-1] if trajectory else None
            if decision.impact_point:
                decision.impact = self._classify_impact(decision.impact_point)
            decision.wicket = WicketResult.UNKNOWN
            decision.reasons.append("Could not project ball to stump zone")

        # ── Step 4: Determine verdict ────────────────────────────────
        decision.verdict = self._determine_verdict(decision)

        return decision

    # ── Pitching Analysis ────────────────────────────────────────────────

    def _find_pitching_point(
        self, trajectory: list[tuple[int, int]]
    ) -> tuple[int, int] | None:
        
        if len(trajectory) < 3:
            return None

        # Look for a bounce: y-velocity was increasing (ball moving down-screen)
        # then decreases (ball bounced up slightly or leveled off)
        best_bounce = None
        best_score = 0

        for i in range(2, len(trajectory) - 1):
            y_prev2 = trajectory[i - 2][1]
            y_prev = trajectory[i - 1][1]
            y_curr = trajectory[i][1]
            y_next = trajectory[i + 1][1]

            dy_before = y_curr - y_prev
            dy_after = y_next - y_curr

            # Ball was going toward batting end (y increasing) then slowed or reversed
            if dy_before > 1 and dy_after < dy_before:
                score = dy_before - dy_after  # bigger change = more likely bounce
                if score > best_score:
                    best_score = score
                    best_bounce = trajectory[i]

        if best_bounce:
            return best_bounce

        # Fallback: use the point with maximum y (deepest point in trajectory)
        if trajectory:
            return max(trajectory, key=lambda p: p[1])
        return None

    def _classify_pitching(self, point: tuple[int, int]) -> PitchingResult:
        """Classify where the ball pitched.
        In line = ball X is within the widest stump/strip corridor."""
        x = point[0]
        left, right = self._get_inline_bounds()

        if x < left:
            return PitchingResult.OUTSIDE_LEG
        elif x > right:
            return PitchingResult.OUTSIDE_OFF
        else:
            return PitchingResult.IN_LINE

    # ── Impact Analysis ──────────────────────────────────────────────────

    def _classify_impact(self, point: tuple[int, int]) -> ImpactResult:
        """Classify where the ball impacted.
        In line = ball X is within the widest stump/strip corridor."""
        x = point[0]
        left, right = self._get_inline_bounds()

        if x > right:
            return ImpactResult.OUTSIDE_OFF
        elif x < left:
            return ImpactResult.OUTSIDE_LEG
        else:
            return ImpactResult.IN_LINE

    def _get_inline_bounds(self) -> tuple[int, int]:
        """Get the widest X boundaries for 'in line' classification.
        Uses the full range of: batting stump edges, bowling stump edges,
        and pitch strip edges — plus a tolerance margin."""
        TOLERANCE = 25  # pixels of margin on each side

        bat_half = Config.BATTING_STUMP_WIDTH // 2
        bowl_half = Config.BOWLING_STUMP_WIDTH // 2

        all_lefts = [
            Config.BATTING_STUMP_X - bat_half,
            Config.BOWLING_STUMP_X - bowl_half,
            Config.PITCH_STRIP_LEFT_TOP,
            Config.PITCH_STRIP_LEFT_BOTTOM,
        ]
        all_rights = [
            Config.BATTING_STUMP_X + bat_half,
            Config.BOWLING_STUMP_X + bowl_half,
            Config.PITCH_STRIP_RIGHT_TOP,
            Config.PITCH_STRIP_RIGHT_BOTTOM,
        ]
        return (min(all_lefts) - TOLERANCE, max(all_rights) + TOLERANCE)

    # ── Wicket Hit Analysis ──────────────────────────────────────────────

    def _find_wicket_point(
        self,
        trajectory: list[tuple[int, int]],
        predicted_path: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Find where the ball ends up relative to the batting stump rectangle.

        Simple logic: look at ALL points (observed + predicted) and find
        the point closest to the stump center-Y that is within the stump
        Y range. If none, use the closest point overall.
        """
        sr = self.stump_rect
        stump_top = sr["top_y"]
        stump_bot = sr["bottom_y"]
        stump_mid_y = (stump_top + stump_bot) // 2

        all_points = trajectory + predicted_path
        if not all_points:
            return None

        # Find the point within the stump Y band closest to center
        best_pt = None
        best_dist = float('inf')
        for pt in all_points:
            if stump_top <= pt[1] <= stump_bot:
                dist = abs(pt[1] - stump_mid_y)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = pt

        if best_pt:
            return best_pt

        # Fallback: closest point to the stump zone
        closest = min(all_points, key=lambda p: abs(p[1] - stump_mid_y))
        if abs(closest[1] - stump_mid_y) < 100:
            return closest

        return None

    def _classify_wicket_hit(self, point: tuple[int, int]) -> WicketResult:
        """Determine if the ball would hit the stumps.
        Simple: if point is within the 4 corners of batting stump rect → Hitting."""
        x, y = point
        sr = self.stump_rect

        x_in = sr["x1"] <= x <= sr["x2"]
        y_in = sr["top_y"] <= y <= sr["bottom_y"]

        if x_in and y_in:
            return WicketResult.HITTING

        # Classify which side it's missing
        if x < sr["x1"]:
            return WicketResult.MISSING_LEG
        elif x > sr["x2"]:
            return WicketResult.MISSING_OFF
        elif y < sr["top_y"]:
            return WicketResult.MISSING_OVER
        else:
            return WicketResult.MISSING_OVER  # below stumps, unlikely

    # ── Final Verdict ────────────────────────────────────────────────────

    def _determine_verdict(self, decision: LBWDecision) -> Verdict:
        """Apply the LBW rules to determine the final verdict."""
        reasons = decision.reasons

        # Rule 1: Pitched outside leg = NOT OUT (always)
        if decision.pitching == PitchingResult.OUTSIDE_LEG:
            reasons.append("Ball pitched outside leg stump — NOT OUT regardless")
            return Verdict.NOT_OUT

        # Rule 2: Impact outside off (while playing a shot) = NOT OUT
        # Note: We can't detect shot playing, so we flag it
        if decision.impact == ImpactResult.OUTSIDE_OFF:
            reasons.append("Impact outside off stump — NOT OUT if playing a shot")
            return Verdict.NOT_OUT

        # Rule 3: Missing stumps = NOT OUT
        if decision.wicket in (
            WicketResult.MISSING_OFF,
            WicketResult.MISSING_LEG,
            WicketResult.MISSING_OVER,
        ):
            reasons.append(f"Ball projected to be {decision.wicket.value}")
            return Verdict.NOT_OUT

        # All conditions met for OUT
        if (
            decision.pitching in (PitchingResult.IN_LINE, PitchingResult.OUTSIDE_OFF)
            and decision.impact == ImpactResult.IN_LINE
            and decision.wicket == WicketResult.HITTING
        ):
            reasons.append("All three conditions satisfied — OUT")
            return Verdict.OUT

        # Edge cases — umpire's call
        if decision.wicket == WicketResult.HITTING:
            reasons.append("Marginal decision — UMPIRE'S CALL")
            return Verdict.UMPIRES_CALL

        reasons.append("Could not determine conclusive verdict")
        return Verdict.INCONCLUSIVE
