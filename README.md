# WHEELS — AI Gesture-Controlled Driving Copilot

**Drive with your hands. Let AI take the wheel.**

WHEELS is a Python-based computer vision project that allows users to control a vehicle in [Slow Roads](https://slowroads.io/) using real-time hand gestures captured through a webcam.

The project combines **OpenCV, MediaPipe, Python, and Windows keyboard automation** to translate hand movements into driving commands.

WHEELS currently supports proportional steering, five-level acceleration, braking, reversing, and gesture-triggered autodrive.

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

- lwft index finger shown and hold to toggle Slow Roads' native autodrive feature.
- A hold timer and cooldown help prevent accidental activation.
- Manual keyboard commands are released while WHEELS is in autodrive mode.



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

### Run WHEELS

```powershell
python main.py
```

Open [Slow Roads](https://slowroads.io/) in your browser.


## Known Limitations and Disadvantages

**The main limitation of the current prototype is steering sensitivity.**

Small changes in the left hand's position can sometimes produce larger-than-expected steering responses, especially during sharp turns or when the hand moves away from its calibrated neutral position.

Although WHEELS uses a dead zone and smoothing, steering is not yet as precise or consistent as a physical steering wheel or analog controller.




Developed by [abdulwasaeee](https://github.com/abdulwasaeee).

## License

See the repository's `LICENSE` file, if included.
