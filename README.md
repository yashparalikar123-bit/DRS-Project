# DRS Project - Decision Review System for Cricket

A complete cricket Decision Review System (DRS) built with Python, YOLOv8, and OpenCV. 
Features Hawk-Eye style ball tracking, trajectory prediction, LBW analysis, pitch map detection, and edge detection.

## Features

- **Ball Tracking** — Real-time ball detection with Kalman filter smoothing
- **Trajectory Prediction** — Polynomial curve fitting for accurate future path prediction
- **LBW Analysis** — Full LBW decision engine (pitching, impact, wicket hitting)
- **Pitch Map** — Visual pitch map showing where the ball pitched and impacted
- **Stump Visualization** — 3D stump zone overlay with hit/miss prediction
- **Edge Detection** — Frame differencing for detecting edges off bat/pad
- **Hawk-Eye Replay** — Slow-motion replay with DRS-style overlays
- **DRS Verdict** — Automated OUT/NOT OUT decision with reasoning

## Setup

### 1. Install Python (macOS)
```bash
# Install Xcode Command Line Tools (includes Python 3)
xcode-select --install

# Or install via Homebrew
brew install python
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Edit the `.env` file and add your Roboflow API key:
```
ROBOFLOW_API_KEY=your_actual_api_key
```

### 5. Run the DRS System
```bash
python main.py
```

Or with custom options:
```bash
python main.py --input video.mp4 --output result.mp4 --show-preview
```

## Project Structure

```
drs-project/
├── main.py                 # Entry point & CLI
├── config.py               # Configuration management
├── ball_tracker.py          # Ball detection & Kalman filter tracking
├── trajectory_predictor.py  # Polynomial trajectory prediction
├── lbw_analyzer.py          # LBW decision engine
├── pitch_map.py             # Pitch map & stump zone rendering
├── edge_detector.py         # Edge detection via frame analysis
├── drs_visualizer.py        # Hawk-Eye style overlays & rendering
├── drs_engine.py            # Main DRS orchestrator
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
└── README.md                # This file
```

## How It Works

1. **Detection**: YOLOv8 model detects the cricket ball in each frame
2. **Tracking**: Kalman filter smooths detection positions and handles missed frames
3. **Trajectory**: Polynomial regression fits a curve to the ball path
4. **Prediction**: Extrapolates the ball path forward to the stump zone
5. **Analysis**: LBW engine checks pitching (in line/outside), impact (in line/outside), and wicket hitting
6. **Verdict**: Renders DRS-style overlay with the final decision

## Configuration

Edit `config.py` to adjust:
- Stump position and dimensions
- Pitch boundaries (good length, full length zones)
- Detection confidence thresholds
- Prediction parameters
- Visual styling (colors, fonts, line thickness)
