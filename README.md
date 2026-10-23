# WHEELS — AI Gesture-Controlled Driving Copilot

**Drive with your hands. Let AI take the wheel.**

WHEELS is a Python-based computer vision project that allows users to control a vehicle in [Slow Roads](https://slowroads.io/) using real-time hand gestures captured through a webcam.

The project combines **OpenCV, MediaPipe, Python, and Windows keyboard automation** to translate hand movements into driving commands.

WHEELS currently supports proportional steering, five-level acceleration, braking, reversing, and gesture-triggered autodrive.

> **Project status:** Working prototype — active development.

## Features

### Hand Gesture Driving

**Right hand — Acceleration and braking**

- Closed fist: Handbrake
- 1–5 extended fingers: Five acceleration levels
- Fingers pointing downward: Reverse

**Left hand — Steering**

- Move left: Steer left
- Move right: Steer right
- Return to calibrated neutral position: Drive straight
- Steering intensity varies with horizontal hand displacement

**Both hands — Autodrive**

- Point both hands downward and hold to toggle Slow Roads' native autodrive feature.
- A hold timer and cooldown help prevent accidental activation.
- Manual keyboard commands are released while WHEELS is in autodrive mode.

### Additional Features

- Real-time hand tracking using MediaPipe
- 21 hand landmarks per detected hand
- Gesture smoothing to reduce jitter
- Adjustable steering sensitivity and dead zone
- Steering calibration
- Live webcam dashboard with throttle, steering, and FPS indicators
- AI-assisted green-screen background replacement
- Automatic release of keyboard controls when hand tracking is lost
- Emergency-stop shortcut

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main application |
| OpenCV | Webcam processing and dashboard |
| MediaPipe | Hand landmark detection and person segmentation |
| NumPy | Numerical processing |
| pynput | Windows keyboard automation |
| Slow Roads | Browser-based driving environment |

## How It Works

1. The webcam captures real-time video.
2. MediaPipe detects hand landmarks.
3. WHEELS identifies the left and right hands.
4. The gesture classifier interprets acceleration and braking gestures.
5. Palm-center tracking calculates steering direction and intensity.
6. The control manager converts gestures into keyboard commands.
7. Slow Roads receives the commands and responds.

The background-removal module processes the displayed webcam preview separately from the original hand-tracking input.

## Installation

### Requirements

- Windows 10 or 11
- Python 3.12
- Working webcam
- Git
- Modern web browser
- Internet connection for Slow Roads

### Clone the Repository

```powershell
git clone https://github.com/abdulwasaeee/WHEELS.git
cd WHEELS
```

### Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Download MediaPipe Models

Create a `models` directory and download the required Hand Landmarker and Selfie Segmenter models.

```powershell
New-Item -ItemType Directory -Force models

Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "models\hand_landmarker.task"

Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite" `
  -OutFile "models\selfie_segmenter.tflite"
```

### Run WHEELS

```powershell
python main.py
```

Open [Slow Roads](https://slowroads.io/) in your browser.

Calibrate steering using `C` in the WHEELS dashboard, focus the game, and press `F8` to enable gesture driving.

## Keyboard Shortcuts

| Key | Function |
|---|---|
| C | Calibrate steering (dashboard focused) |
| F8 | Enable or disable WHEELS |
| Esc | Emergency stop and exit |
| Q | Quit from the dashboard window |

## Known Limitations and Disadvantages

### 1. Sensitive Steering Calibration

**The main limitation of the current prototype is steering sensitivity.**

Small changes in the left hand's position can sometimes produce larger-than-expected steering responses, especially during sharp turns or when the hand moves away from its calibrated neutral position.

Although WHEELS uses a dead zone and smoothing, steering is not yet as precise or consistent as a physical steering wheel or analog controller.

**Planned improvements:**

- Adaptive steering sensitivity
- Improved neutral-position calibration
- Nonlinear steering response curves
- Better filtering of hand movement
- Speed-dependent steering sensitivity
- More accurate low-intensity steering control

### 2. Digital Keyboard Steering

Slow Roads receives simulated keyboard inputs rather than true analog steering commands.

WHEELS approximates proportional steering using timed key presses, which can produce uneven turning at low steering intensities.

### 3. Webcam Tracking Limitations

Performance depends on lighting, hand visibility, camera positioning, and tracking stability.

Occlusion or sudden hand movements can temporarily interrupt gesture recognition.

### 4. Background Segmentation Artifacts

The green-screen feature may produce imperfect edges around hands and fingers. Small areas of the original background may remain visible.

### 5. Native Autodrive State

WHEELS toggles Slow Roads' built-in autodrive using a keyboard command. It does not independently verify the game's internal autodrive state.

## Future Improvements

- [ ] Improve steering calibration and sensitivity
- [ ] Implement adaptive steering curves
- [ ] Optimize gesture recognition latency
- [ ] Improve green-screen segmentation
- [ ] Add voice assistant: "Hey Wheels"
- [ ] Add conversational AI responses
- [ ] Integrate music and real-world weather
- [ ] Implement voice-controlled driving takeover
- [ ] Explore custom vision-based autonomous driving
- [ ] Add performance benchmarks and demonstration videos

## Project Scope

WHEELS is an experimental computer vision and AI project intended for browser-based driving simulation.

It is **not designed to control real vehicles**.

## Author

Developed by [abdulwasaeee](https://github.com/abdulwasaeee).

## License

See the repository's `LICENSE` file, if included.