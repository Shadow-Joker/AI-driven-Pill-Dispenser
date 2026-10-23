import cv2
import time
import math
import mediapipe as mp

from flask import Flask, Response
from picamera2 import Picamera2


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20

FACE_MODEL_PATH = (
    "/home/raspberrypi/pill_dispenser/"
    "models/mediapipe/face_landmarker.task"
)

MOUTH_OPEN_THRESHOLD = 0.30

app = Flask(__name__)


# ============================================================
# CAMERA
# ============================================================

print("\nStarting patient camera...\n")

picam2 = Picamera2(1)

camera_config = picam2.create_video_configuration(
    main={
        "size": (
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        ),
        "format": "RGB888"
    },
    controls={
        "FrameRate": CAMERA_FPS
    }
)

picam2.configure(camera_config)
picam2.start()

time.sleep(2)

print("Patient camera started.\n")


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

face_landmarker = FaceLandmarker.create_from_options(
    options
)

print("MediaPipe Face Landmarker started.\n")


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ============================================================
# MOUTH OPENING RATIO
#
# MediaPipe Face Mesh landmark indices:
#
# 13  = upper inner lip
# 14  = lower inner lip
#
# 78  = left mouth region
# 308 = right mouth region
# ============================================================

def calculate_mouth_ratio(landmarks):

    vertical_distance = distance(
        landmarks[13],
        landmarks[14]
    )

    horizontal_distance = distance(
        landmarks[78],
        landmarks[308]
    )

    if horizontal_distance <= 0:
        return 0.0

    return vertical_distance / horizontal_distance


# ============================================================
# DRAW MOUTH
# ============================================================

MOUTH_POINTS = [

    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375,
    291,

    78, 95, 88, 178, 87,
    14, 317, 402, 318, 324,
    308
]


def draw_mouth(frame, landmarks):

    height, width, _ = frame.shape

    for index in MOUTH_POINTS:

        x = int(
            landmarks[index].x * width
        )

        y = int(
            landmarks[index].y * height
        )

        cv2.circle(
            frame,
            (x, y),
            2,
            (0, 255, 0),
            -1
        )


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    timestamp_ms = 0

    while True:

        # Capture RGB frame
        frame = picam2.capture_array()

        # MediaPipe expects SRGB/RGB image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        timestamp_ms += int(
            1000 / CAMERA_FPS
        )

        # Detect face landmarks
        result = face_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        display_frame = frame.copy()

        # Convert RGB -> BGR for OpenCV
        display_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_RGB2BGR
        )

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            mouth_ratio = calculate_mouth_ratio(
                landmarks
            )

            draw_mouth(
                display_frame,
                landmarks
            )

            # ------------------------------------------------
            # DETERMINE MOUTH STATE
            # ------------------------------------------------

            if mouth_ratio >= MOUTH_OPEN_THRESHOLD:

                mouth_state = "MOUTH OPEN"
                color = (0, 0, 255)

            else:

                mouth_state = "MOUTH CLOSED"
                color = (0, 255, 0)

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            cv2.putText(
                display_frame,
                mouth_state,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                display_frame,
                f"Ratio: {mouth_ratio:.3f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                display_frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Encode JPEG
        success, buffer = cv2.imencode(
            ".jpg",
            display_frame
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
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

        <title>MediaPipe Mouth Detection</title>

        <style>

            body {
                background: #111;
                color: white;
                text-align: center;
                font-family: Arial;
            }

            img {
                margin-top: 20px;
                max-width: 95%;
                border: 2px solid #444;
            }

        </style>

    </head>

    <body>

        <h1>MediaPipe Mouth Detection Test</h1>

        <img src="/video_feed">

    </body>

    </html>
    """


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
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 40)
    print("MEDIAPIPE MOUTH DETECTION TEST")
    print("=" * 40)

    print("\nOpen in browser:\n")
    print("http://RASPBERRY_PI_IP:5000\n")

    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            debug=False
        )

    finally:

        print("\nStopping application...")

        picam2.stop()

        face_landmarker.close()
