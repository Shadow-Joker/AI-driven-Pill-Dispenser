import os
import time
import atexit
import signal

import cv2
import mediapipe as mp
from flask import Flask, Response
from picamera2 import Picamera2

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.expanduser(
    "~/pill_dispenser/models/mediapipe/hand_landmarker.task"
)

CAMERA_INDEX = 1

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MAX_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

running = True
camera = None
hand_landmarker = None


# ============================================================
# START CAMERA
# ============================================================

print("\nStarting patient camera...\n")

camera = Picamera2(CAMERA_INDEX)

camera_config = camera.create_video_configuration(
    main={
        "size": (FRAME_WIDTH, FRAME_HEIGHT),
        "format": "RGB888"
    }
)

camera.configure(camera_config)
camera.start()

time.sleep(2)

print("Patient camera started.")


# ============================================================
# LOAD MEDIAPIPE HAND LANDMARKER
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nHand model not found:\n{MODEL_PATH}\n"
    )

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

hand_landmarker = vision.HandLandmarker.create_from_options(
    options
)

print("MediaPipe Hand Landmarker started.")


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand_landmarks(frame, result):

    height, width, _ = frame.shape

    hand_count = len(result.hand_landmarks)

    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        # Convert normalized landmarks to pixel coordinates
        points = []

        for landmark in hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        # Draw connections
        for start, end in HAND_CONNECTIONS:

            if start < len(points) and end < len(points):

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

        # Draw landmarks
        for i, point in enumerate(points):

            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1
            )

        # Get handedness
        hand_label = "Hand"
        confidence = 0.0

        if hand_index < len(result.handedness):

            category = result.handedness[hand_index][0]

            hand_label = category.category_name
            confidence = category.score

        # Display hand label near wrist
        if len(points) > 0:

            wrist_x, wrist_y = points[0]

            text = (
                f"{hand_label} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (wrist_x, max(30, wrist_y - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display number of detected hands
    cv2.putText(
        frame,
        f"Hands Detected: {hand_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return frame


# ============================================================
# MJPEG VIDEO STREAM
# ============================================================

def generate_frames():

    global running

    timestamp_ms = 0

    while running:

        try:

            # Capture frame from Picamera2
            frame = camera.capture_array()

            # Ensure RGB
            if frame is None:
                continue

            # MediaPipe expects SRGB image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame
            )

            # Generate strictly increasing timestamp
            timestamp_ms += 33

            # Detect hand landmarks
            result = hand_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Convert RGB to BGR for OpenCV drawing/JPEG
            display_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            # Draw landmarks
            display_frame = draw_hand_landmarks(
                display_frame,
                result
            )

            # Encode JPEG
            success, buffer = cv2.imencode(
                ".jpg",
                display_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    85
                ]
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

        except GeneratorExit:
            break

        except Exception as e:

            if running:
                print(
                    f"Frame processing error: {e}"
                )

            time.sleep(0.05)


# ============================================================
# WEB PAGE
# ============================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>

    <head>

        <title>MediaPipe Hand Detection</title>

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 30px;
            }

            h1 {
                margin-bottom: 10px;
            }

            .container {
                display: inline-block;
                background: #222;
                padding: 15px;
                border-radius: 12px;
            }

            img {
                width: 640px;
                max-width: 100%;
                border-radius: 8px;
                border: 2px solid #444;
            }

            p {
                color: #aaa;
            }

        </style>

    </head>

    <body>

        <h1>MediaPipe Hand Detection</h1>

        <p>
            Patient Camera — Live Hand Landmark Detection
        </p>

        <div class="container">

            <img
                src="/video_feed"
            >

        </div>

    </body>

    </html>
    """


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    global running

    running = False

    print("\nStopping application...")

    try:

        if hand_landmarker is not None:
            hand_landmarker.close()

    except Exception as e:

        print(
            f"Hand landmarker cleanup error: {e}"
        )

    try:

        if camera is not None:
            camera.stop()

    except Exception as e:

        print(
            f"Camera cleanup error: {e}"
        )


atexit.register(cleanup)


def signal_handler(sig, frame):

    cleanup()

    os._exit(0)


signal.signal(
    signal.SIGINT,
    signal_handler
)

signal.signal(
    signal.SIGTERM,
    signal_handler
)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n====================================")
    print("MEDIAPIPE HAND DETECTION TEST")
    print("====================================")

    print("\nOpen in browser:\n")
    print("http://RASPBERRY_PI_IP:5000\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=False,
        use_reloader=False
    )
