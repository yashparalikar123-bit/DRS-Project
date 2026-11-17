"""Diagnose impact point issue."""
from config import Config
from ball_tracker import BallTracker
from trajectory_predictor import TrajectoryPredictor
from inference import get_model
import supervision as sv
import cv2

model = get_model(model_id=Config.MODEL_ID)
cap = cv2.VideoCapture(Config.INPUT_VIDEO)
w, h = int(cap.get(3)), int(cap.get(4))
tracker = BallTracker(w, h)
idx = 0
while True:
    ret, f = cap.read()
    if not ret:
        break
    r = model.infer(f, confidence=Config.CONFIDENCE_THRESHOLD)[0]
    d = sv.Detections.from_inference(r)
    tracker.update(d, idx)
    idx += 1
cap.release()

traj = tracker.get_trajectory()
bat_top = Config.BATTING_STUMP_TOP_Y
bat_bot = Config.BATTING_STUMP_BOTTOM_Y
strip_left = min(Config.PITCH_STRIP_LEFT_TOP, Config.PITCH_STRIP_LEFT_BOTTOM)
strip_right = max(Config.PITCH_STRIP_RIGHT_TOP, Config.PITCH_STRIP_RIGHT_BOTTOM)
stump_x1 = Config.BATTING_STUMP_X - Config.BATTING_STUMP_WIDTH // 2
stump_x2 = Config.BATTING_STUMP_X + Config.BATTING_STUMP_WIDTH // 2

print(f"Total trajectory points: {len(traj)}")
print(f"Impact point (last): {traj[-1]}")
print(f"Batting stump Y: {bat_top} to {bat_bot}")
print(f"Strip X: {strip_left} to {strip_right}")
print(f"Stump X: {stump_x1} to {stump_x2}")
print()

# Points within batting stump Y range
in_zone = [(x, y) for x, y in traj if bat_top <= y <= bat_bot]
print(f"Points in stump Y zone: {len(in_zone)}")
if in_zone:
    zx = [p[0] for p in in_zone]
    print(f"  X range: {min(zx)} to {max(zx)}")
    print(f"  First: {in_zone[0]}")
    print(f"  Last:  {in_zone[-1]}")
    print(f"  Mid:   {in_zone[len(in_zone)//2]}")

# Show every 10th point
print("\nTrajectory sample (every 10th):")
for i in range(0, len(traj), 10):
    x, y = traj[i]
    on_strip = strip_left <= x <= strip_right
    in_stump_y = bat_top <= y <= bat_bot
    print(f"  [{i:3d}] ({x:4d}, {y:3d})  strip={'YES' if on_strip else 'no '}  stumpY={'YES' if in_stump_y else 'no '}")
