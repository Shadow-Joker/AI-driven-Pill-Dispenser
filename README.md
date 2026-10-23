# AI-driven Pill Dispenser

An intelligent, state-controlled automated pill dispensing system built for embedded healthcare assistance (Raspberry Pi). The system integrates computer vision for patient facial recognition, hand/mouth gesture landmark tracking, pill tray verification, and interactive web dashboard management.

---

## 🌟 Key Features

- **State-Controlled Dispensing Engine:** Deterministic state machine managing patient authentication, tray release, intake verification, and status monitoring.
- **Biometric Patient Verification:**
  - Face detection using **YuNet** (`face_detection_yunet_2023mar.onnx`).
  - Face recognition and embedding extraction using **SFace** (`face_recognition_sface_2021dec.onnx`).
- **Medication Intake Tracking:**
  - Hand and facial landmark tracking via **MediaPipe** (`hand_landmarker.task`, `face_landmarker.task`).
  - Real-time pill detection using **NCNN** optimized models.
  - Computer vision-based tray occupancy verification (empty vs. loaded).
- **Web Interface & Dashboard:**
  - Real-time video feeds with visual overlay of detection states.
  - Patient schedule management and dispensing logs.
  - Interactive manual override and calibration dashboard.
- **Hardware Integration:**
  - Camera streaming via **Picamera2** / OpenCV.
  - GPIO actuator and servo control for physical pill release via **RPi.GPIO**.

---

## 📂 Repository Structure

```
├── app_state_controlled.py     # Main Flask server and dispensing state controller
├── Hardware Details/           # Schematics and bill of materials
│   ├── Circuit Connections.pdf # Wiring diagrams and GPIO pinout connections
│   └── List of Components.pdf  # Detailed BOM and component specifications
├── data/                       # Schedule configurations, patient records, and reference masks
│   ├── medication_logs.json
│   ├── medication_schedule.json
│   ├── patients.json
│   └── empty_tray_reference.png
├── models/                     # Deep learning models and weights
│   ├── face_recognition/       # YuNet & SFace ONNX models
│   ├── mediapipe/              # MediaPipe landmark tasks
│   └── pill_detector_ncnn/     # NCNN model params and weights
├── pill_dataset/               # Dataset samples (with_pill / no_pill)
├── static/                     # Web UI frontend assets (CSS, JavaScript)
├── templates/                  # Web dashboard HTML templates
└── test codes/                 # Hardware & model test scripts
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Target Platform:** Raspberry Pi 4 / 5 running Raspberry Pi OS (or Linux/macOS for development/simulation).
- **Python:** 3.9+

### 2. Dependencies

Install core dependencies:
```bash
pip install -r requirements.txt
```

> **Note on Raspberry Pi Hardware Packages:**  
> `picamera2` and `RPi.GPIO` are pre-installed or installable via apt on Raspberry Pi OS:
> ```bash
> sudo apt install -y python3-picamera2 python3-rpi.gpio
> ```

### 3. Run the Application

```bash
python app_state_controlled.py
```

Access the dashboard by navigating to `http://localhost:5000` (or `http://<raspberrypi-ip>:5000`) in your web browser.
