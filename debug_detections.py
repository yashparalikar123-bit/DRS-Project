"""Debug script: inspect model detections on a few frames."""
from __future__ import annotations
import cv2
from dotenv import load_dotenv
load_dotenv()

from inference import get_model
import supervision as sv

model = get_model(model_id="drs-5gvgi/3")
cap = cv2.VideoCapture("v16_crop.mp4")

for frame_idx in range(0, 156, 15):  # sample every 15 frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # Try with explicit low confidence
    results = model.infer(frame, confidence=0.05)[0]
    
    print(f"\n--- Frame {frame_idx} ---")
    print(f"  Result type: {type(results).__name__}")
    print(f"  Result attrs: {[a for a in dir(results) if not a.startswith('_')][:20]}")
    
    if hasattr(results, 'predictions'):
        preds = results.predictions
        print(f"  Predictions count: {len(preds)}")
        for j, p in enumerate(preds[:5]):
            print(f"    [{j}] {p}")
    
    detections = sv.Detections.from_inference(results)
    n = len(detections.xyxy)
    print(f"  Supervision detections: {n}")
    for i in range(min(n, 5)):
        box = detections.xyxy[i]
        cid = detections.class_id[i] if detections.class_id is not None else "N/A"
        conf = detections.confidence[i] if detections.confidence is not None else "N/A"
        data = detections.data if hasattr(detections, 'data') else {}
        class_name = ""
        if 'class_name' in data:
            class_name = data['class_name'][i]
        print(f"    [{i}] class_id={cid}  name='{class_name}'  conf={conf:.3f}  box={box}")

cap.release()
print("\nDone.")
