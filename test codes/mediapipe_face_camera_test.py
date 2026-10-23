import time
import cv2
import mediapipe as mp

from flask import Flask, Response
from picamera2 import Picamera2

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CAMERA SETUP
# Camera 0 = Patient Camera
# ============================================================

print("Starting patient camera...")

picam = Picamera2(1)

config = picam.create_preview_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    }
)

picam.configure(config)
picam.start()

time.sleep(2)

print("Patient camera started.")


# ============================================================
# MEDIAPIPE FACE DETECTOR
# ============================================================

MODEL_PATH = "models/blaze_face_short_range.tflite"

options = vision.FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.5
)

face_detector = vision.FaceDetector.create_from_options(
    options
)

print("MediaPipe Face Detector started.")


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    timestamp_ms = 0

    while True:

        # Capture RGB frame
        frame = picam.capture_array()

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        # Timestamp must continuously increase
        timestamp_ms += 33

        # MediaPipe face detection
        result = face_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Convert RGB to BGR for OpenCV drawing/JPEG
        display_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        face_count = len(result.detections)

        # ----------------------------------------------------
        # DRAW FACE BOXES
        # ----------------------------------------------------

        for detection in result.detections:

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            # Prevent invalid coordinates
            x = max(0, x)
            y = max(0, y)

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            confidence = 0.0

            if detection.categories:
                confidence = detection.categories[0].score

            cv2.putText(
                display_frame,
                f"Face: {confidence:.2f}",
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            f"Faces Detected: {face_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # ENCODE JPEG
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            display_frame
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>

    <head>

        <title>MediaPipe Face Detection</title>

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial;
                text-align: center;
                margin: 0;
                padding: 30px;
            }

            h1 {
                color: #00ff88;
            }

            .camera-container {
                display: inline-block;
                background: #222;
                padding: 15px;
                border-radius: 10px;
            }

            img {
                width: 640px;
                max-width: 100%;
                border-radius: 5px;
                border: 2px solid #00ff88;
            }

        </style>

    </head>

    <body>

        <h1>MediaPipe Face Detection</h1>

        <div class="camera-container">

            <img src="/video_feed">

        </div>

    </body>

    </html>
    """


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("")
    print("====================================")
    print("MEDIAPIPE FACE DETECTION TEST")
    print("====================================")
    print("")
    print("Open in browser:")
    print("")
    print("http://RASPBERRY_PI_IP:5000")
    print("")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=False
    )
