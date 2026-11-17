import os
from dotenv import load_dotenv
import cv2
import numpy as np
from inference import get_model
import supervision as sv

# Load environment variables from .env file
load_dotenv()  # Ensure your API key is set in the .env file

# Load a pre-trained YOLOv8 model
model = get_model(model_id="drs-5gvgi/3")

# Define input and output video file paths
input_video_file = "v16_crop.mp4"  # Path to your input video file
output_video_file = "aqwn.mp4"  # Path to save the annotated output video

# Open the input video
cap = cv2.VideoCapture(input_video_file)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the middle 40% vertical strip range
left_bound = int(width * 0.3)
right_bound = int(width * 0.7)

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_file, fourcc, fps*0.3, (width, height))

# Create Supervision annotators
bounding_box_annotator = sv.BoxAnnotator()  # Updated to BoxAnnotator
label_annotator = sv.LabelAnnotator()

# Initialize list to track the ball's trajectory
ball_trajectory = []

# Number of future frames to predict at the end
num_future_frames = 8
scaling_factor = 0.5  # Scale down the velocity to shorten the future path
num_points_for_prediction = 5  # Number of last points to use for velocity calculation

# Process the video frame by frame
last_frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frames are left

    # Save the last frame
    last_frame = frame.copy()

    # Run inference on the current frame
    results = model.infer(frame)[0]

    # Load the results into the Supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Annotate the frame with inference results
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Extract the ball's bounding box and track its position within the middle 40% strip
    for i in range(len(detections.xyxy)):  # Use len(detections.xyxy) to iterate over detections
        x1, y1, x2, y2 = detections.xyxy[i]  # Get bounding box coordinates
        class_id = detections.class_id[i]  # Get class ID
        if class_id == 0:  # Assuming class_id 0 corresponds to the ball
            # Check if the ball is within the middle 40% vertical strip
            x_center = int((x1 + x2) / 2)  # Calculate center x coordinate
            y_center = int((y1 + y2) / 2)  # Calculate center y coordinate
            if left_bound <= x_center <= right_bound:
                # Store the ball's position
                ball_trajectory.append((x_center, y_center))

                # Draw a circle on the detected ball
                cv2.circle(annotated_frame, (x_center, y_center), 5, (0, 255, 0), -1)

    # Draw the solid pipe-like trajectory of the ball
    if len(ball_trajectory) > 1:
        # Draw circles at trajectory points to create a pipe-like appearance
        for point in ball_trajectory:
            cv2.circle(annotated_frame, point, 8, (0, 0, 255), -1)  # Red circles for trajectory points

        # Draw solid lines for the trajectory
        cv2.polylines(annotated_frame, [np.array(ball_trajectory)], False, (255, 0, 0), thickness=4)  # Blue color, thicker line

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Predict the future path at the end using the last frame and draw it on the last frame
if last_frame is not None and len(ball_trajectory) > num_points_for_prediction:
    # Calculate the average velocity using the last few points
    total_delta_x = 0
    total_delta_y = 0
    count = 0
    for i in range(-num_points_for_prediction, -1):
        delta_x = ball_trajectory[i + 1][0] - ball_trajectory[i][0]
        delta_y = ball_trajectory[i + 1][1] - ball_trajectory[i][1]
        total_delta_x += delta_x
        total_delta_y += delta_y
        count += 1
    avg_delta_x = (total_delta_x / count) * scaling_factor
    avg_delta_y = (total_delta_y / count) * scaling_factor

    # Get the last point
    x_last, y_last = ball_trajectory[-1]
    future_path = []

    # Generate future points based on the average velocity
    for i in range(1, num_future_frames + 1):
        future_x = int(x_last + i * avg_delta_x)
        future_y = int(y_last + i * avg_delta_y)
        future_path.append((future_x, future_y))

    # Draw the actual trajectory on the last frame with a thicker, brighter line
    for point in ball_trajectory:
        cv2.circle(last_frame, point, 8, (0, 0, 255), -1)  # Red circles for actual trajectory points
    cv2.polylines(last_frame, [np.array(ball_trajectory)], False, (0, 0, 255), thickness=6)  # Thicker red line for actual trajectory

    # Draw the predicted future path in a different color (e.g., bright yellow)
    for point in future_path:
        cv2.circle(last_frame, point, 8, (0, 255, 255), -1)  # Yellow circles for future path
    cv2.polylines(last_frame, [np.array(future_path)], False, (0, 255, 255), thickness=4)  # Yellow line for future path

    # Hold the last frame with the future path for some time
    for _ in range(int(fps * 2)):  # Display the last frame for 2 seconds
        out.write(last_frame)

# Release resources
cap.release()
out.release()

print(f"Output video saved as {output_video_file}")