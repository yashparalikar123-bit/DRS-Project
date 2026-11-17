"""Quick test: draw stumps and strip on a single frame and save as image."""
from __future__ import annotations
import cv2
from dotenv import load_dotenv
load_dotenv()

from config import Config
from drs_visualizer import DRSVisualizer

cap = cv2.VideoCapture("v16_crop.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read frame")
    exit()

h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")

viz = DRSVisualizer(w, h)

# Draw stumps directly on frame
frame = viz.draw_stumps(frame, highlight="neutral")
frame = viz.draw_drs_label(frame)

# Also draw a big test rectangle to confirm drawing works at all
cv2.rectangle(frame, (100, 100), (300, 200), (0, 0, 255), 3)
cv2.putText(frame, "TEST OVERLAY", (110, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite("test_frame.jpg", frame)
print("Saved test_frame.jpg — open it to check if stumps/strip are visible")
print(f"Batting stumps: {Config.get_batting_stump_rect()}")
print(f"Bowling stumps: {Config.get_bowling_stump_rect()}")
