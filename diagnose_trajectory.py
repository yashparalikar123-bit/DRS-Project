"""Diagnose trajectory to understand LBW verdict."""
import cv2
import numpy as np
from inference import get_model
import supervision as sv
from config import Config
from ball_tracker import BallTracker
from trajectory_predictor import TrajectoryPredictor
from lbw_analyzer import LBWAnalyzer

model = get_model(model_id=Config.MODEL_ID)
cap = cv2.VideoCapture(Config.INPUT_VIDEO)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tracker = BallTracker(w, h)

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.infer(frame, confidence=Config.CONFIDENCE_THRESHOLD)[0]
    dets = sv.Detections.from_inference(results)
    tracker.update(dets, idx)
    idx += 1
cap.release()

traj = tracker.get_trajectory()
print(f"Trajectory: {len(traj)} points")
if traj:
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    print(f"X range: {min(xs)} to {max(xs)}")
    print(f"Y range: {min(ys)} to {max(ys)}")
    print(f"First 5: {traj[:5]}")
    print(f"Last  5: {traj[-5:]}")
    print(f"Batting stump Y: {Config.BATTING_STUMP_TOP_Y} to {Config.BATTING_STUMP_BOTTOM_Y}")
    print(f"Bowling stump Y: {Config.BOWLING_STUMP_TOP_Y} to {Config.BOWLING_STUMP_BOTTOM_Y}")

    # Predict future path
    pred = TrajectoryPredictor()
    pred.fit(traj)
    predicted = pred.predict_future(traj)
    if predicted:
        pxs = [p[0] for p in predicted]
        pys = [p[1] for p in predicted]
        print(f"\nPredicted: {len(predicted)} points")
        print(f"Predicted X range: {min(pxs)} to {max(pxs)}")
        print(f"Predicted Y range: {min(pys)} to {max(pys)}")
        print(f"Predicted first 5: {predicted[:5]}")
        print(f"Predicted last  5: {predicted[-5:]}")

        # Check stump mid Y 
        stump_mid_y = (Config.BATTING_STUMP_TOP_Y + Config.BATTING_STUMP_BOTTOM_Y) // 2
        print(f"\nStump mid Y: {stump_mid_y}")

        # Check if any predicted point crosses stump mid
        for i, pt in enumerate(predicted):
            if pt[1] >= stump_mid_y:
                print(f"Predicted point {i} reaches stump mid Y: {pt}")
                break
        else:
            print(f"NO predicted point reaches stump mid Y={stump_mid_y}")

        # Check the LBW analyzer
        analyzer = LBWAnalyzer(w, h)
        decision = analyzer.analyze(traj, predicted)
        print(f"\n{decision.summary()}")
    else:
        print("\nNo predicted path generated!")
