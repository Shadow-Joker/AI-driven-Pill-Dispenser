import os
import json
import time
import base64
import threading
from datetime import datetime

import cv2
import ncnn
import numpy as np
import mediapipe as mp

# OpenCV Zoo face recognition uses YuNet + SFace.
# YuNet detects the face and provides the 5 landmarks needed by SFace;
# SFace converts the aligned face into an embedding vector.

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    render_template,
)

from picamera2 import Picamera2
import RPi.GPIO as GPIO


# ============================================================
# FLASK
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(BASE_DIR, "data")

PATIENTS_FILE = os.path.join(
    DATA_DIR,
    "patients.json"
)

MEDICATION_LOG_FILE = os.path.join(
    DATA_DIR,
    "medication_logs.json"
)

MEDICATION_SCHEDULE_FILE = os.path.join(
    DATA_DIR,
    "medication_schedule.json"
)

FACE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "mediapipe",
    "face_landmarker.task",
)

HAND_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "mediapipe",
    "hand_landmarker.task",
)

FACE_RECOGNITION_DIR = os.path.join(
    BASE_DIR,
    "models",
    "face_recognition",
)

YUNET_MODEL = os.path.join(
    FACE_RECOGNITION_DIR,
    "face_detection_yunet_2023mar.onnx",
)

SFACE_MODEL = os.path.join(
    FACE_RECOGNITION_DIR,
    "face_recognition_sface_2021dec.onnx",
)

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20

# Pill camera uses higher resolution for better YOLO input.
PILL_CAMERA_WIDTH = 1536
PILL_CAMERA_HEIGHT = 864

JPEG_QUALITY = 80


# ============================================================
# CAMERA GLOBALS
# ============================================================

patient_camera = None
pill_camera = None

patient_camera_lock = threading.Lock()
pill_camera_lock = threading.Lock()


# ============================================================
# PILL / TRAY DETECTION CONFIGURATION
# ============================================================

REFERENCE_PATH = os.path.join(
    DATA_DIR,
    "empty_tray_reference.png"
)

REFERENCE_MASK_PATH = os.path.join(
    DATA_DIR,
    "empty_tray_mask.png"
)

NORMALIZED_WIDTH = 420
NORMALIZED_HEIGHT = 300

# Tray detection

MIN_TRAY_AREA = 2500
MIN_TRAY_WIDTH = 60
MIN_TRAY_HEIGHT = 40

# Pill detection

MIN_PILL_AREA = 100
MAX_PILL_AREA = 1500
MAX_PILL_AREA_RATIO = 0.20

MIN_PILL_WIDTH = 8
MIN_PILL_HEIGHT = 8

MIN_SOLIDITY = 0.55

PILL_DARK_VALUE_MAX = 85

PILL_COLOR_SATURATION_MIN = 45

PILL_COLOR_VALUE_MAX = 220

MIN_PILL_COLOR_RATIO = 0.30

# Number of consecutive frames required
# before declaring a pill as detected.

REQUIRED_DETECTION_FRAMES = 2

REQUIRED_REMOVAL_FRAMES = 3

# ============================================================
# PILL DETECTION GLOBALS
# ============================================================

reference_image = None
reference_mask = None

pill_tray_detected = False
pill_detected = False

pill_tray_status = "WAITING"
pill_status = "WAITING"

pill_area = 0.0
pill_bbox = None

pill_detection_counter = 0
pill_count = 0
pill_confidences = []
pill_removal_counter = 0

# Pill count captured when a medication cycle verifies the dispensed pill(s).
# Used so pickup can be detected even if other pills remain in the tray.
medication_pill_count_at_verification = 0

# YOLO NCNN pill detector
PILL_YOLO_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "pill_detector_ncnn"
)
PILL_YOLO_INPUT_SIZE = 640
PILL_YOLO_CONF_THRESHOLD = 0.25
PILL_YOLO_NMS_THRESHOLD = 0.45
PILL_YOLO_MIN_AREA = 100
PILL_YOLO_MAX_AREA = 1500
PILL_YOLO_MIN_WIDTH = 8
PILL_YOLO_MIN_HEIGHT = 8
PILL_YOLO_MIN_SOLIDITY = 0.55
PILL_YOLO_ANALYSIS_WIDTH = 420
PILL_YOLO_ANALYSIS_HEIGHT = 300

pill_yolo_net = None
pill_yolo_lock = threading.Lock()

# Debug images

last_difference = None
last_binary = None

# ============================================================
# MEDIAPIPE GLOBALS
# ============================================================

face_landmarker = None
hand_landmarker = None

# Face recognition globals
face_detector_yn = None
face_recognizer_sf = None
face_recognition_lock = threading.Lock()

RECOGNITION_COSINE_THRESHOLD = 0.363
RECOGNITION_EVERY_N_FRAMES = 5
recognition_frame_counter = 0
recognized_patient_id = None
recognized_patient_name = None
recognized_face_score = 0.0
recognized_face_status = "NO FACE"

mediapipe_lock = threading.Lock()

# Timestamp required for MediaPipe VIDEO mode
hand_timestamp_ms = 0


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Little finger
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17),
]

# ============================================================
# DETECTION STATUS
# ============================================================

detection_status = {
    "face_count": 0,
    "face_detected": False,
    "mouth_open": False,
    "mouth_ratio": 0.0,
    "hand_detected": False,
    "hand_count": 0,
}


# ============================================================
# MEDICATION STATE MACHINE
# ============================================================

MEDICATION_IDLE = "IDLE"

MEDICATION_WAITING_FOR_PATIENT = (
    "WAITING_FOR_PATIENT"
)

MEDICATION_DISPENSING = (
    "DISPENSING"
)

MEDICATION_WAITING_FOR_PILL = (
    "WAITING_FOR_PILL"
)

MEDICATION_WAITING_FOR_PICKUP = (
    "WAITING_FOR_PICKUP"
)

MEDICATION_WAITING_FOR_HAND = (
    "WAITING_FOR_HAND"
)

MEDICATION_WAITING_FOR_MOUTH = (
    "WAITING_FOR_MOUTH"
)

MEDICATION_TAKEN = (
    "MEDICATION_TAKEN"
)

MEDICATION_TIMEOUT = (
    "TIMEOUT"
)


# ------------------------------------------------------------
# CURRENT STATE
# ------------------------------------------------------------

medication_state = MEDICATION_IDLE


# ------------------------------------------------------------
# STATE MACHINE DATA
# ------------------------------------------------------------

medication_active = False

medication_pill_number = None

medication_start_time = None

medication_state_time = None

medication_patient_id = None

medication_schedule_id = None

# ------------------------------------------------------------
# TIMEOUT SETTINGS
# ------------------------------------------------------------

PATIENT_TIMEOUT = 60

PILL_DETECTION_TIMEOUT = 15

PICKUP_TIMEOUT = 60

HAND_TIMEOUT = 30

MOUTH_TIMEOUT = 30


# ------------------------------------------------------------
# STATE MACHINE LOCK
# ------------------------------------------------------------

medication_lock = threading.Lock()

# ============================================================
# PILL DISPENSER MOTORS
# TB6612FNG
# ============================================================

# ------------------------------------------------------------
# Motor 1 = Pill 1
# ------------------------------------------------------------

PILL1_AIN1 = 17
PILL1_AIN2 = 27
PILL1_PWM = 18


# ------------------------------------------------------------
# Motor 2 = Pill 2
# ------------------------------------------------------------

PILL2_BIN1 = 22
PILL2_BIN2 = 23
PILL2_PWM = 13


# ------------------------------------------------------------
# TB6612FNG STANDBY
# ------------------------------------------------------------

MOTOR_STBY = 24


# ============================================================
# BUZZER
# ============================================================

BUZZER_PIN = 26

# ============================================================
# MOTOR SETTINGS
# ============================================================

MOTOR_PWM_FREQUENCY = 1000

DEFAULT_MOTOR_SPEED = 75

DEFAULT_DISPENSE_TIME = 4


# ============================================================
# GPIO INITIALIZATION
# ============================================================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

motor_pins = [
    PILL1_AIN1,
    PILL1_AIN2,
    PILL1_PWM,
    PILL2_BIN1,
    PILL2_BIN2,
    PILL2_PWM,
    MOTOR_STBY,
]

for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)


# ============================================================
# BUZZER INITIALIZATION
# ============================================================

GPIO.setup(
    BUZZER_PIN,
    GPIO.OUT
)

GPIO.output(
    BUZZER_PIN,
    GPIO.LOW
)

# Keep motors stopped initially

GPIO.output(PILL1_AIN1, GPIO.LOW)
GPIO.output(PILL1_AIN2, GPIO.LOW)

GPIO.output(PILL2_BIN1, GPIO.LOW)
GPIO.output(PILL2_BIN2, GPIO.LOW)

GPIO.output(MOTOR_STBY, GPIO.LOW)


# PWM

pill1_pwm = GPIO.PWM(
    PILL1_PWM,
    MOTOR_PWM_FREQUENCY,
)

pill2_pwm = GPIO.PWM(
    PILL2_PWM,
    MOTOR_PWM_FREQUENCY,
)

pill1_pwm.start(0)
pill2_pwm.start(0)


# ============================================================
# MOTOR LOCK
# ============================================================

motor_lock = threading.Lock()


# ============================================================
# BUZZER FUNCTIONS
# ============================================================

buzzer_lock = threading.Lock()


def buzzer_beep(
    duration=0.2
):

    with buzzer_lock:

        GPIO.output(
            BUZZER_PIN,
            GPIO.HIGH
        )

        time.sleep(
            duration
        )

        GPIO.output(
            BUZZER_PIN,
            GPIO.LOW
        )


def buzzer_pattern(
    count=3,
    on_time=0.2,
    off_time=0.2
):

    for _ in range(count):

        buzzer_beep(
            on_time
        )

        time.sleep(
            off_time
        )


def start_buzzer_pattern(
    count=3,
    on_time=0.2,
    off_time=0.2
):

    thread = threading.Thread(
        target=buzzer_pattern,
        args=(
            count,
            on_time,
            off_time
        ),
        daemon=True
    )

    thread.start()

# ============================================================
# MEDICATION LOGGING
# ============================================================

medication_log_lock = threading.Lock()


def load_medication_logs():

    if not os.path.exists(
        MEDICATION_LOG_FILE
    ):

        return []


    try:

        with open(
            MEDICATION_LOG_FILE,
            "r"
        ) as file:

            logs = json.load(
                file
            )


        if isinstance(
            logs,
            list
        ):

            return logs


        return []


    except Exception as error:

        print(
            "Error loading medication logs:",
            error
        )

        return []


def save_medication_logs(
    logs
):

    try:

        with open(
            MEDICATION_LOG_FILE,
            "w"
        ) as file:

            json.dump(
                logs,
                file,
                indent=4
            )


        return True


    except Exception as error:

        print(
            "Error saving medication logs:",
            error
        )

        return False


def add_medication_log(
    event,
    result=None,
    message=None
):

    global medication_pill_number
    global medication_patient_id
    global medication_schedule_id


    log_entry = {

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "pill_number":
            medication_pill_number,

        "patient_id":
            medication_patient_id,

        "schedule_id":
            medication_schedule_id,

        "event":
            event,

        "result":
            result,

        "message":
            message,

        "state":
            medication_state
    }


    with medication_log_lock:

        logs = load_medication_logs()

        logs.append(
            log_entry
        )

        # Keep latest 1000 logs only

        logs = logs[-1000:]

        save_medication_logs(
            logs
        )


    print(
        "MEDICATION LOG:",
        log_entry
    )

# ============================================================
# MEDICATION SCHEDULER
# ============================================================

scheduler_running = False

scheduler_lock = threading.Lock()

scheduler_last_trigger = {}


def load_medication_schedule():

    if not os.path.exists(
        MEDICATION_SCHEDULE_FILE
    ):

        return []


    try:

        with open(
            MEDICATION_SCHEDULE_FILE,
            "r"
        ) as file:

            schedules = json.load(
                file
            )


        if isinstance(
            schedules,
            list
        ):

            return schedules


        return []


    except Exception as error:

        print(
            "Error loading medication schedule:",
            error
        )

        return []


def save_medication_schedule(
    schedules
):

    try:

        with open(
            MEDICATION_SCHEDULE_FILE,
            "w"
        ) as file:

            json.dump(
                schedules,
                file,
                indent=4
            )


        return True


    except Exception as error:

        print(
            "Error saving medication schedule:",
            error
        )

        return False


def medication_scheduler():

    global scheduler_running
    global scheduler_last_trigger
    global medication_patient_id
    global medication_schedule_id


    print(
        "MEDICATION SCHEDULER STARTED"
    )


    while scheduler_running:

        try:

            now = datetime.now()

            current_time = now.strftime(
                "%H:%M"
            )

            current_date = now.strftime(
                "%Y-%m-%d"
            )


            schedules = (
                load_medication_schedule()
            )


            for schedule in schedules:


                if not schedule.get(
                    "enabled",
                    True
                ):

                    continue


                schedule_id = str(
                    schedule.get(
                        "schedule_id",
                        ""
                    )
                )


                pill_number = schedule.get(
                    "pill_number"
                )


                schedule_time = str(
                    schedule.get(
                        "time",
                        ""
                    )
                )


                patient_id = schedule.get(
                    "patient_id",
                    None
                )


                if not schedule_id:

                    continue


                if pill_number not in [
                    1,
                    2
                ]:

                    continue


                if current_time != schedule_time:

                    continue


                trigger_key = (
                    f"{schedule_id}_"
                    f"{current_date}_"
                    f"{current_time}"
                )


                if (
                    scheduler_last_trigger.get(
                        schedule_id
                    )
                    ==
                    trigger_key
                ):

                    continue


                if medication_active:

                    print(
                        "Scheduled medication skipped. "
                        "Another medication cycle is active."
                    )

                    continue


                print("")
                print(
                    "=" * 50
                )
                print(
                    "SCHEDULED MEDICATION TRIGGERED"
                )
                print(
                    f"Schedule ID: {schedule_id}"
                )
                print(
                    f"Pill: {pill_number}"
                )
                print(
                    f"Patient: {patient_id}"
                )
                print(
                    "=" * 50
                )


                scheduler_last_trigger[
                    schedule_id
                ] = trigger_key


                medication_patient_id = (
                    patient_id
                )

                medication_schedule_id = (
                    schedule_id
                )


                add_medication_log(
                    event="SCHEDULE_TRIGGERED",
                    result="STARTED",
                    message=(
                        f"Scheduled medication "
                        f"triggered at "
                        f"{current_time}"
                    )
                )


                # Alert patient

                start_buzzer_pattern(
                    count=5,
                    on_time=0.3,
                    off_time=0.3
                )


                success, message = (
                    start_medication_cycle(
                        pill_number,
                        patient_id=patient_id,
                        schedule_id=schedule_id
                    )
                )


                if not success:

                    add_medication_log(
                        event="SCHEDULE_START_FAILED",
                        result="FAILED",
                        message=message
                    )


        except Exception as error:

            print(
                "Medication scheduler error:",
                error
            )


        time.sleep(
            1
        )


def start_medication_scheduler():

    global scheduler_running


    if scheduler_running:

        return


    scheduler_running = True


    thread = threading.Thread(
        target=medication_scheduler,
        daemon=True
    )

    thread.start()


# ============================================================
# LOAD PATIENT DATABASE
# ============================================================

def load_patients():

    if not os.path.exists(PATIENTS_FILE):

        with open(PATIENTS_FILE, "w") as file:

            json.dump([], file, indent=4)

        return []

    try:

        with open(PATIENTS_FILE, "r") as file:

            data = json.load(file)

        if isinstance(data, list):

            return data

        return []

    except Exception as error:

        print("Error loading patients:", error)

        return []


# ============================================================
# SAVE PATIENT DATABASE
# ============================================================

def save_patients(patients):

    try:

        with open(PATIENTS_FILE, "w") as file:

            json.dump(
                patients,
                file,
                indent=4,
            )

        return True

    except Exception as error:

        print("Error saving patients:", error)

        return False


# ============================================================
# INITIALIZE PATIENT FILE
# ============================================================

load_patients()


def load_pill_yolo_model():

    global pill_yolo_net

    param_path = os.path.join(
        PILL_YOLO_MODEL_DIR,
        "model.ncnn.param"
    )

    bin_path = os.path.join(
        PILL_YOLO_MODEL_DIR,
        "model.ncnn.bin"
    )

    if not os.path.exists(param_path):
        raise RuntimeError(
            f"YOLO NCNN param not found:\n{param_path}"
        )

    if not os.path.exists(bin_path):
        raise RuntimeError(
            f"YOLO NCNN model not found:\n{bin_path}"
        )

    print("\nLoading YOLO NCNN pill detector...")

    net = ncnn.Net()

    ret = net.load_param(param_path)
    if ret != 0:
        raise RuntimeError(
            f"Failed to load YOLO NCNN param: {ret}"
        )

    ret = net.load_model(bin_path)
    if ret != 0:
        raise RuntimeError(
            f"Failed to load YOLO NCNN model: {ret}"
        )

    pill_yolo_net = net

    print("YOLO NCNN pill detector loaded successfully.")


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def start_cameras():

    global patient_camera
    global pill_camera

    print("\nStarting cameras...\n")

    try:

        patient_camera = Picamera2(0)

        patient_config = patient_camera.create_video_configuration(

            main={
                "size": (
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT,
                ),
                "format": "RGB888",
            },

            controls={
                "FrameRate": CAMERA_FPS,
            },
        )

        patient_camera.configure(patient_config)

        patient_camera.start()

        print("Patient camera started.")

    except Exception as error:

        print("ERROR starting patient camera:", error)

        patient_camera = None


    try:

        pill_camera = Picamera2(1)

        pill_config = pill_camera.create_video_configuration(

            main={
                "size": (
                    PILL_CAMERA_WIDTH,
                    PILL_CAMERA_HEIGHT,
                ),
                "format": "RGB888",
            },

            controls={
                "FrameRate": CAMERA_FPS,
            },
        )

        pill_camera.configure(pill_config)

        pill_camera.start()

        print(
            f"Pill camera started at {PILL_CAMERA_WIDTH}x{PILL_CAMERA_HEIGHT}."
        )

    except Exception as error:

        print("ERROR starting pill camera:", error)

        pill_camera = None


    time.sleep(2)



# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

def initialize_mediapipe():

    global face_landmarker
    global hand_landmarker

    print("\nInitializing MediaPipe...\n")

    # --------------------------------------------------------
    # FACE LANDMARKER
    # --------------------------------------------------------

    if not os.path.exists(FACE_MODEL):

        raise RuntimeError(
            f"Face model not found:\n{FACE_MODEL}"
        )

    face_options = mp.tasks.vision.FaceLandmarkerOptions(

        base_options=mp.tasks.BaseOptions(
            model_asset_path=FACE_MODEL
        ),

        running_mode=mp.tasks.vision.RunningMode.IMAGE,

        num_faces=2,

        min_face_detection_confidence=0.5,

        min_face_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )

    face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        face_options
    )

    print("MediaPipe Face Landmarker started.")


    # --------------------------------------------------------
    # HAND LANDMARKER
    # --------------------------------------------------------

    if not os.path.exists(HAND_MODEL):

        raise RuntimeError(
            f"Hand model not found:\n{HAND_MODEL}"
        )

    hand_options = mp.tasks.vision.HandLandmarkerOptions(

        base_options=mp.tasks.BaseOptions(
            model_asset_path=HAND_MODEL
        ),

        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        num_hands=2,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )

    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        hand_options
    )

    print("MediaPipe Hand Landmarker started.")


# ============================================================
# INITIALIZE FACE RECOGNITION
# YuNet + SFace (OpenCV Zoo)
# ============================================================

def initialize_face_recognition():

    global face_detector_yn
    global face_recognizer_sf

    if not os.path.exists(YUNET_MODEL):
        raise RuntimeError(
            "YuNet model not found:\n" + YUNET_MODEL
        )

    if not os.path.exists(SFACE_MODEL):
        raise RuntimeError(
            "SFace model not found:\n" + SFACE_MODEL
        )

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "This OpenCV build does not provide FaceDetectorYN. "
            "Install/update OpenCV with FaceDetectorYN support."
        )

    if not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError(
            "This OpenCV build does not provide FaceRecognizerSF. "
            "Install/update OpenCV with FaceRecognizerSF support."
        )

    face_detector_yn = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (320, 320),
        0.90,
        0.30,
        5000,
    )

    face_recognizer_sf = cv2.FaceRecognizerSF.create(
        SFACE_MODEL,
        "",
    )

    print("OpenCV YuNet + SFace face recognition started.")


def decode_patient_image(image_data):

    if not image_data:
        return None

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data)
        array = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception as error:
        print("Patient image decode error:", error)
        return None


def detect_faces_yunet(image_bgr, top_k=10):

    if face_detector_yn is None or image_bgr is None:
        return None

    height, width = image_bgr.shape[:2]

    with face_recognition_lock:
        face_detector_yn.setInputSize((width, height))
        face_detector_yn.setTopK(top_k)
        _, faces = face_detector_yn.detect(image_bgr)

    return faces


def get_face_embedding(image_bgr, face_row):

    if face_recognizer_sf is None:
        return None

    with face_recognition_lock:
        aligned = face_recognizer_sf.alignCrop(
            image_bgr,
            face_row,
        )
        feature = face_recognizer_sf.feature(aligned)

    return np.asarray(feature, dtype=np.float32).flatten()


def migrate_patient_embeddings():

    patients = load_patients()
    changed = False

    for patient in patients:

        if patient.get("face_embedding"):
            continue

        image_data = patient.get("image")
        image_bgr = decode_patient_image(image_data)

        if image_bgr is None:
            print(
                f"Could not decode stored image for "
                f"patient {patient.get('patient_id')}"
            )
            continue

        embedding, error = create_face_embedding(image_bgr)

        if error:
            print(
                f"Could not create embedding for "
                f"patient {patient.get('patient_id')}: {error}"
            )
            continue

        patient["face_embedding"] = embedding
        changed = True

        print(
            f"Face embedding created for "
            f"patient {patient.get('patient_id')}"
        )

    if changed:
        save_patients(patients)


def create_face_embedding(image_bgr):

    faces = detect_faces_yunet(image_bgr, top_k=10)

    if faces is None or len(faces) == 0:
        return None, "No face detected in registration image."

    if len(faces) != 1:
        return None, "Registration image must contain exactly one face."

    embedding = get_face_embedding(
        image_bgr,
        faces[0],
    )

    if embedding is None or embedding.size == 0:
        return None, "Could not generate face embedding."

    return embedding.tolist(), None


def recognize_patient(image_bgr):

    global recognized_patient_id
    global recognized_patient_name
    global recognized_face_score
    global recognized_face_status

    faces = detect_faces_yunet(image_bgr, top_k=10)

    if faces is None or len(faces) == 0:
        recognized_patient_id = None
        recognized_patient_name = None
        recognized_face_score = 0.0
        recognized_face_status = "NO FACE"
        return None, None, 0.0

    if len(faces) > 1:
        recognized_patient_id = None
        recognized_patient_name = None
        recognized_face_score = 0.0
        recognized_face_status = "MULTIPLE FACES"
        return None, None, 0.0

    embedding = get_face_embedding(
        image_bgr,
        faces[0],
    )

    if embedding is None:
        recognized_patient_id = None
        recognized_patient_name = None
        recognized_face_score = 0.0
        recognized_face_status = "EMBEDDING ERROR"
        return None, None, 0.0

    best_patient = None
    best_score = -1.0

    for patient in load_patients():

        stored = patient.get("face_embedding")

        if not stored:
            continue

        try:
            stored_vector = np.asarray(
                stored,
                dtype=np.float32,
            ).reshape(1, -1)

            query_vector = embedding.reshape(1, -1)

            score = float(
                face_recognizer_sf.match(
                    stored_vector,
                    query_vector,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )

            if score > best_score:
                best_score = score
                best_patient = patient

        except Exception as error:
            print(
                "Face comparison error:",
                error,
            )

    if (
        best_patient is not None
        and best_score >= RECOGNITION_COSINE_THRESHOLD
    ):
        recognized_patient_id = str(
            best_patient.get("patient_id")
        )
        recognized_patient_name = str(
            best_patient.get("name") or "Unknown"
        )
        recognized_face_score = best_score
        recognized_face_status = "RECOGNIZED"

        return (
            recognized_patient_id,
            recognized_patient_name,
            best_score,
        )

    recognized_patient_id = None
    recognized_patient_name = None
    recognized_face_score = max(0.0, best_score)
    recognized_face_status = "UNKNOWN"

    return None, None, max(0.0, best_score)


# ============================================================
# GET PATIENT CAMERA FRAME
# ============================================================

def get_patient_frame():

    if patient_camera is None:

        return None

    try:

        with patient_camera_lock:

            frame = patient_camera.capture_array()

        return frame

    except Exception as error:

        print("Patient camera error:", error)

        return None


# ============================================================
# GET PILL CAMERA FRAME
# ============================================================

def get_pill_frame():

    if pill_camera is None:

        return None

    try:

        with pill_camera_lock:

            frame = pill_camera.capture_array()

        return frame

    except Exception as error:

        print("Pill camera error:", error)

        return None


# ============================================================
# PILL TRAY DETECTION
#
# Dynamic tray detection.
# No fixed X/Y coordinates are used.
# ============================================================


# ============================================================
# CREATE PILL DISPENSING ZONE MASK
#
# The detected contour may contain the complete dispenser.
# We only want the upper narrow / semicircular dispensing area.
#
# This is detected dynamically by looking for the point where
# the contour suddenly becomes much wider and transitions into
# the rectangular dispenser body.
# ============================================================

def create_pill_zone_mask(
    contour,
    frame_shape
):

    height, width = frame_shape[:2]

    # --------------------------------------------------------
    # Full detected dispenser mask
    # --------------------------------------------------------

    full_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.drawContours(
        full_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # --------------------------------------------------------
    # Get bounding rectangle
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        contour
    )

    if w <= 0 or h <= 0:

        return full_mask

    # --------------------------------------------------------
    # Measure contour width at every horizontal row
    # --------------------------------------------------------

    row_widths = []

    for yy in range(
        y,
        min(
            y + h,
            height
        )
    ):

        xs = np.where(
            full_mask[yy] > 0
        )[0]

        if len(xs) < 2:

            row_widths.append(
                0
            )

        else:

            row_widths.append(
                int(
                    xs[-1] - xs[0] + 1
                )
            )

    if len(row_widths) == 0:

        return full_mask

    max_row_width = max(
        row_widths
    )

    if max_row_width <= 0:

        return full_mask

    # --------------------------------------------------------
    # Smooth row widths slightly
    # --------------------------------------------------------

    widths_array = np.array(
        row_widths,
        dtype=np.float32
    )

    widths_smooth = cv2.GaussianBlur(
        widths_array.reshape(
            -1,
            1
        ),
        (
            1,
            9
        ),
        0
    ).flatten()

    # --------------------------------------------------------
    # Find where the narrow upper section transitions into
    # the wide rectangular dispenser body.
    #
    # When width reaches approximately 82% of the maximum
    # width and remains wide for several rows, that is treated
    # as the bottom of the pill dispensing zone.
    # --------------------------------------------------------

    transition_width = (
        max_row_width * 0.82
    )

    split_index = None

    required_rows = max(
        5,
        int(
            h * 0.03
        )
    )

    for i in range(
        len(widths_smooth)
    ):

        end_index = min(
            len(widths_smooth),
            i + required_rows
        )

        section = widths_smooth[
            i:end_index
        ]

        if len(section) < required_rows:

            continue

        if np.all(
            section >= transition_width
        ):

            split_index = i

            break

    # --------------------------------------------------------
    # Fallback if transition is not found
    # --------------------------------------------------------

    if split_index is None:

        split_index = int(
            len(row_widths) * 0.60
        )

    # --------------------------------------------------------
    # Convert to actual image Y coordinate
    # --------------------------------------------------------

    split_y = (
        y +
        split_index
    )

    # Small margin so the lower part of the semicircle is not
    # accidentally removed.

    split_y += int(
        h * 0.03
    )

    split_y = min(
        split_y,
        height
    )

    # --------------------------------------------------------
    # Keep only upper dispensing region
    # --------------------------------------------------------

    pill_zone_mask = full_mask.copy()

    pill_zone_mask[
        split_y:,
        :
    ] = 0

    # --------------------------------------------------------
    # Remove unstable outer edges
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            5,
            5
        )
    )

    pill_zone_mask = cv2.erode(
        pill_zone_mask,
        kernel,
        iterations=1
    )

    return pill_zone_mask

def detect_tray(frame):
    if frame is None:
        return None, None

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    normalized = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    _, binary = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_TRAY_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_TRAY_WIDTH or h < MIN_TRAY_HEIGHT:
            continue

        if area > width * height * 0.90:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        solidity = area / hull_area

        center_y = y + h / 2.0
        vertical_score = 1.0 - min(1.0, center_y / float(height))

        score = (
            area * 0.55
            + solidity * 5000.0
            + vertical_score * 2500.0
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None, None

    full_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.drawContours(
        full_mask,
        [best_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    x, y, w, h = cv2.boundingRect(best_contour)

    # Use the COMPLETE dynamically detected tray/dispenser contour
    # as the allowed region. Do not impose a fixed vertical cutoff:
    # the pill can appear anywhere inside the actual tray bowl.
    tray_mask = full_mask.copy()

    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    tray_mask = cv2.erode(
        tray_mask,
        edge_kernel,
        iterations=1
    )

    if cv2.countNonZero(tray_mask) < 500:
        return None, None

    return best_contour, tray_mask


# ============================================================
# EXTRACT NORMALIZED TRAY
# ============================================================

def extract_normalized_tray(
    frame,
    contour,
    mask
):

    if contour is None:

        return None, None


    if mask is None:

        return None, None


    x, y, w, h = cv2.boundingRect(
        contour
    )


    # Small dynamic margin

    margin = max(
        3,
        int(
            min(w, h) * 0.02
        )
    )


    x1 = max(
        0,
        x - margin
    )


    y1 = max(
        0,
        y - margin
    )


    x2 = min(
        frame.shape[1],
        x + w + margin
    )


    y2 = min(
        frame.shape[0],
        y + h + margin
    )


    tray = frame[
        y1:y2,
        x1:x2
    ].copy()


    tray_mask = mask[
        y1:y2,
        x1:x2
    ].copy()


    if tray.size == 0:

        return None, None


    if tray_mask.size == 0:

        return None, None


    # Normalize tray size

    tray = cv2.resize(
        tray,
        (
            NORMALIZED_WIDTH,
            NORMALIZED_HEIGHT
        ),
        interpolation=cv2.INTER_LINEAR
    )


    tray_mask = cv2.resize(
        tray_mask,
        (
            NORMALIZED_WIDTH,
            NORMALIZED_HEIGHT
        ),
        interpolation=cv2.INTER_NEAREST
    )


    _, tray_mask = cv2.threshold(
        tray_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )


    return (
        tray,
        tray_mask
    )


# ============================================================
# LOAD EMPTY TRAY REFERENCE
# ============================================================

def load_empty_tray_reference():

    global reference_image
    global reference_mask


    if not os.path.exists(
        REFERENCE_PATH
    ):

        print(
            "No empty tray reference found."
        )

        return False


    if not os.path.exists(
        REFERENCE_MASK_PATH
    ):

        print(
            "No empty tray mask found."
        )

        return False


    image = cv2.imread(
        REFERENCE_PATH
    )


    mask = cv2.imread(
        REFERENCE_MASK_PATH,
        cv2.IMREAD_GRAYSCALE
    )


    if image is None:

        print(
            "Could not load empty tray reference."
        )

        return False


    if mask is None:

        print(
            "Could not load empty tray mask."
        )

        return False


    reference_image = image

    reference_mask = mask


    print(
        "Empty tray reference loaded successfully."
    )


    return True


# ============================================================
# CAPTURE EMPTY TRAY REFERENCE
# ============================================================

def capture_empty_tray_reference():

    global reference_image
    global reference_mask


    # --------------------------------------------------------
    # Capture directly from pill camera
    # --------------------------------------------------------

    frame_rgb = get_pill_frame()


    if frame_rgb is None:

        return (
            False,
            "Pill camera frame unavailable."
        )


    # Picamera gives RGB

    frame = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2BGR
    )


    contour, mask = detect_tray(
        frame
    )


    if contour is None:

        return (
            False,
            "Tray not found. Adjust camera and try again."
        )


    tray, tray_mask = extract_normalized_tray(
        frame,
        contour,
        mask
    )


    if tray is None:

        return (
            False,
            "Could not extract tray."
        )


    # Save reference

    reference_image = tray.copy()

    reference_mask = tray_mask.copy()


    cv2.imwrite(
        REFERENCE_PATH,
        reference_image
    )


    cv2.imwrite(
        REFERENCE_MASK_PATH,
        reference_mask
    )


    print()

    print(
        "=" * 50
    )

    print(
        "EMPTY TRAY REFERENCE SAVED"
    )

    print(
        "=" * 50
    )

    print(
        REFERENCE_PATH
    )

    print()


    return (
        True,
        "Empty tray reference captured successfully."
    )


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_gray(
    image,
    mask
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )


    values = gray[
        mask > 0
    ]


    if len(values) < 20:

        return gray


    low = np.percentile(
        values,
        2
    )


    high = np.percentile(
        values,
        98
    )


    if high <= low + 1:

        return gray


    normalized = (

        (
            gray.astype(
                np.float32
            )

            -

            low
        )

        *

        255.0

        /

        (
            high -
            low
        )

    )


    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(
        np.uint8
    )


    return normalized


# ============================================================
# PILL DETECTION
#
# METHOD A:
# Compare current tray with empty tray.
#
# METHOD B:
# Detect dark pill-like object.
#
# Both are combined.
# ============================================================

def letterbox_pill_image(image, new_size=640):

    height, width = image.shape[:2]

    scale = min(
        new_size / float(width),
        new_size / float(height)
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (new_size, new_size, 3),
        114,
        dtype=np.uint8
    )

    pad_x = (new_size - new_width) // 2
    pad_y = (new_size - new_height) // 2

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    return canvas, scale, pad_x, pad_y

def extract_yolo_tray_crop(frame, contour, mask):

    if contour is None or mask is None:
        return None, None, None, None

    x, y, w, h = cv2.boundingRect(contour)

    margin = max(5, int(min(w, h) * 0.02))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)

    tray = frame[y1:y2, x1:x2].copy()
    tray_mask = mask[y1:y2, x1:x2].copy()

    if tray.size == 0 or tray_mask.size == 0:
        return None, None, None, None

    analysis_tray = cv2.resize(
        tray,
        (PILL_YOLO_ANALYSIS_WIDTH, PILL_YOLO_ANALYSIS_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    analysis_mask = cv2.resize(
        tray_mask,
        (PILL_YOLO_ANALYSIS_WIDTH, PILL_YOLO_ANALYSIS_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )

    _, analysis_mask = cv2.threshold(
        analysis_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return tray, analysis_tray, analysis_mask, (x1, y1, x2, y2)

def estimate_yolo_pill_area(tray, tray_mask, bbox):

    x, y, w, h = bbox

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(tray.shape[1], x + w)
    y2 = min(tray.shape[0], y + h)

    if x2 <= x1 or y2 <= y1:
        return 0.0, None

    crop = tray[y1:y2, x1:x2]
    crop_mask = tray_mask[y1:y2, x1:x2]

    if crop.size == 0 or cv2.countNonZero(crop_mask) == 0:
        return 0.0, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    foreground = (
        (gray < 220)
        | (
            (hsv[:, :, 1] >= PILL_COLOR_SATURATION_MIN)
            & (hsv[:, :, 2] <= PILL_COLOR_VALUE_MAX)
        )
    ).astype(np.uint8) * 255

    foreground = cv2.bitwise_and(
        foreground,
        crop_mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0, None

    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))

    if area <= 0:
        return 0.0, None

    return area, best


def detect_pill(frame, contour, mask):
    """Run YOLO only inside the dynamically detected tray and return all valid pills."""

    global pill_yolo_net
    global pill_confidences

    if pill_yolo_net is None:
        return False, 0.0, None, 0, []

    tray, analysis_tray, analysis_mask, crop_box = extract_yolo_tray_crop(
        frame, contour, mask
    )

    if tray is None:
        return False, 0.0, None, 0, []

    image, scale, pad_x, pad_y = letterbox_pill_image(
        tray,
        PILL_YOLO_INPUT_SIZE
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mat = ncnn.Mat.from_pixels(
        rgb,
        ncnn.Mat.PixelType.PIXEL_RGB,
        PILL_YOLO_INPUT_SIZE,
        PILL_YOLO_INPUT_SIZE
    )

    mat.substract_mean_normalize(
        [],
        [1.0 / 255.0] * 3
    )

    try:
        with pill_yolo_lock:
            with pill_yolo_net.create_extractor() as ex:
                ret = ex.input("in0", mat)
                if ret != 0:
                    return False, 0.0, None, 0, []

                ret, out = ex.extract("out0")
                if ret != 0:
                    return False, 0.0, None, 0, []
    except Exception as error:
        print("YOLO inference error:", error)
        return False, 0.0, None, 0, []

    output = np.array(out)

    if output.ndim != 2 or output.shape[0] < 5:
        return False, 0.0, None, 0, []

    boxes = []
    scores = []

    for i in range(output.shape[1]):
        score = float(output[4, i])

        if not np.isfinite(score) or score < PILL_YOLO_CONF_THRESHOLD:
            continue

        x_center = float(output[0, i])
        y_center = float(output[1, i])
        width = float(output[2, i])
        height = float(output[3, i])

        if not all(np.isfinite(v) for v in (x_center, y_center, width, height)):
            continue

        if width <= 0 or height <= 0:
            continue

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0

        boxes.append([x1, y1, width, height])
        scores.append(score)

    if not boxes:
        pill_confidences = []
        return False, 0.0, None, 0, []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        PILL_YOLO_CONF_THRESHOLD,
        PILL_YOLO_NMS_THRESHOLD
    )

    if len(indices) == 0:
        pill_confidences = []
        return False, 0.0, None, 0, []

    candidates = []

    for raw_index in indices:
        index = int(raw_index)

        x1, y1, box_width, box_height = boxes[index]
        confidence = scores[index]
        x2 = x1 + box_width
        y2 = y1 + box_height

        nx1 = (x1 - pad_x) / scale
        ny1 = (y1 - pad_y) / scale
        nx2 = (x2 - pad_x) / scale
        ny2 = (y2 - pad_y) / scale

        nx1 = max(0, min(tray.shape[1] - 1, int(nx1)))
        ny1 = max(0, min(tray.shape[0] - 1, int(ny1)))
        nx2 = max(0, min(tray.shape[1] - 1, int(nx2)))
        ny2 = max(0, min(tray.shape[0] - 1, int(ny2)))

        nw = nx2 - nx1
        nh = ny2 - ny1

        if nw < PILL_YOLO_MIN_WIDTH or nh < PILL_YOLO_MIN_HEIGHT:
            continue

        center_x = int((nx1 + nx2) / 2)
        center_y = int((ny1 + ny2) / 2)

        frame_center_x = crop_box[0] + center_x
        frame_center_y = crop_box[1] + center_y

        if not (
            0 <= frame_center_x < frame.shape[1]
            and 0 <= frame_center_y < frame.shape[0]
            and mask[frame_center_y, frame_center_x] > 0
        ):
            continue

        sx = PILL_YOLO_ANALYSIS_WIDTH / float(tray.shape[1])
        sy = PILL_YOLO_ANALYSIS_HEIGHT / float(tray.shape[0])

        ax1 = int(nx1 * sx)
        ay1 = int(ny1 * sy)
        ax2 = int(nx2 * sx)
        ay2 = int(ny2 * sy)

        estimated_area, area_contour = estimate_yolo_pill_area(
            analysis_tray,
            analysis_mask,
            (ax1, ay1, ax2 - ax1, ay2 - ay1)
        )

        if estimated_area > 0:
            if estimated_area < PILL_YOLO_MIN_AREA:
                continue

            if estimated_area > PILL_YOLO_MAX_AREA:
                continue

            hull = cv2.convexHull(area_contour)
            hull_area = cv2.contourArea(hull)

            if hull_area <= 0:
                continue

            solidity = estimated_area / hull_area
            if solidity < PILL_YOLO_MIN_SOLIDITY:
                continue
        else:
            estimated_area = 0.0

        candidates.append({
            "confidence": confidence,
            "bbox": (nx1, ny1, nw, nh),
            "frame_bbox": (
                int(crop_box[0] + nx1),
                int(crop_box[1] + ny1),
                int(nw),
                int(nh)
            ),
            "area": estimated_area
        })

    if not candidates:
        pill_confidences = []
        return False, 0.0, None, 0, []

    candidates.sort(
        key=lambda c: c["confidence"],
        reverse=True
    )

    pill_confidences = [
        float(c["confidence"])
        for c in candidates
    ]

    crop_x1, crop_y1, _, _ = crop_box

    best = candidates[0]

    best_bbox = (
        int(crop_x1 + best["bbox"][0]),
        int(crop_y1 + best["bbox"][1]),
        int(best["bbox"][2]),
        int(best["bbox"][3])
    )

    return (
        True,
        float(best["area"]),
        best_bbox,
        len(candidates),
        candidates
    )



# ============================================================
# MAP PILL BOX BACK TO ORIGINAL CAMERA FRAME
# ============================================================

def map_pill_box_to_frame(
    contour,
    normalized_box
):

    if normalized_box is None:

        return None


    x, y, w, h = cv2.boundingRect(
        contour
    )


    nx, ny, nw, nh = normalized_box


    scale_x = (
        w /
        float(NORMALIZED_WIDTH)
    )


    scale_y = (
        h /
        float(NORMALIZED_HEIGHT)
    )


    return (

        int(
            x +
            nx * scale_x
        ),

        int(
            y +
            ny * scale_y
        ),

        max(
            1,
            int(
                nw * scale_x
            )
        ),

        max(
            1,
            int(
                nh * scale_y
            )
        )

    )


# ============================================================
# PROCESS PILL FRAME
# ============================================================

def patient_ai_should_run():

    return medication_state in [

        MEDICATION_WAITING_FOR_PATIENT,
        MEDICATION_WAITING_FOR_HAND,
        MEDICATION_WAITING_FOR_MOUTH,

    ]


def patient_face_ai_should_run():

    return medication_state in [

        MEDICATION_WAITING_FOR_PATIENT,
        MEDICATION_WAITING_FOR_MOUTH,

    ]


def patient_hand_ai_should_run():

    return medication_state in [

        MEDICATION_WAITING_FOR_HAND,
        MEDICATION_WAITING_FOR_MOUTH,

    ]


def patient_recognition_should_run():

    return medication_state == MEDICATION_WAITING_FOR_PATIENT


def pill_ai_should_run():

    return medication_state in [

        MEDICATION_WAITING_FOR_PILL,
        MEDICATION_WAITING_FOR_PICKUP,

    ]


def process_pill_frame(frame):

    global pill_tray_detected
    global pill_detected
    global pill_tray_status
    global pill_status
    global pill_area
    global pill_bbox
    global pill_count
    global pill_confidences
    global pill_detection_counter
    global pill_removal_counter

    output = frame.copy()

    # --------------------------------------------------------
    # STATE-CONTROLLED AI
    #
    # Pill/tray detection is only needed after dispensing and
    # while waiting for the pill to be picked up. During all
    # other states, return the live camera frame without
    # running OpenCV tray processing or YOLO.
    # --------------------------------------------------------
    if not pill_ai_should_run():

        pill_tray_detected = False
        pill_detected = False
        pill_tray_status = "AI STANDBY"
        pill_status = "WAITING FOR MEDICATION"
        pill_area = 0.0
        pill_bbox = None
        pill_count = 0
        pill_confidences = []
        pill_detection_counter = 0

        cv2.putText(
            output,
            "AI STANDBY",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            output,
            "PILL DETECTION OFF",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 165, 0),
            2
        )

        return output

    contour, tray_mask = detect_tray(frame)

    if contour is None:
        pill_tray_detected = False
        pill_detected = False
        pill_tray_status = "TRAY NOT FOUND"
        pill_status = "WAITING FOR TRAY"
        pill_area = 0.0
        pill_bbox = None
        pill_count = 0
        pill_confidences = []
        pill_detection_counter = 0

        cv2.putText(
            output,
            "TRAY NOT FOUND",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        return output

    pill_tray_detected = True
    pill_tray_status = "TRAY FOUND"

    # Run YOLO on the clean camera frame first. This prevents the
    # cyan tray outline from becoming part of the YOLO input crop.
    detected, area, bbox, detected_count, candidates = detect_pill(
        frame,
        contour,
        tray_mask
    )

    # Draw every accepted YOLO pill candidate on the output frame.
    for number, candidate in enumerate(candidates, start=1):
        frame_x1, frame_y1, nw, nh = candidate["frame_bbox"]
        frame_x2 = frame_x1 + nw
        frame_y2 = frame_y1 + nh

        cv2.rectangle(
            output,
            (frame_x1, frame_y1),
            (frame_x2, frame_y2),
            (0, 255, 0),
            3
        )

        label = f"PILL {number}: {candidate['confidence']:.2f}"
        if candidate["area"] > 0:
            label += f" A:{candidate['area']:.0f}"
        else:
            label += " A:N/A"

        cv2.putText(
            output,
            label,
            (frame_x1, max(25, frame_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Draw the dynamic tray outline after YOLO inference.
    mask_contours, _ = cv2.findContours(
        tray_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if mask_contours:
        largest_mask_contour = max(
            mask_contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            output,
            [largest_mask_contour],
            -1,
            (0, 255, 255),
            3
        )

    if detected and detected_count > 0:
        pill_detection_counter += 1
        pill_removal_counter = 0
    else:
        pill_detection_counter = 0
        pill_removal_counter += 1

    stable_detection = (
        pill_detection_counter >= REQUIRED_DETECTION_FRAMES
    )

    if stable_detection:
        pill_detected = True
        pill_status = "PILL DETECTED"
        pill_area = area
        pill_bbox = bbox
        pill_count = detected_count
        pill_removal_counter = 0

    else:
        if medication_state == MEDICATION_WAITING_FOR_PICKUP:
            # Keep the last confirmed pill state during
            # temporary YOLO misses.
            pill_removal_counter += 1

            if pill_removal_counter >= REQUIRED_REMOVAL_FRAMES:
                pill_detected = False
                pill_status = "TRAY EMPTY"
                pill_area = 0.0
                pill_bbox = None
                pill_count = 0

        else:
            pill_detected = False
            pill_status = "TRAY EMPTY"
            pill_area = 0.0
            pill_bbox = None
            pill_count = 0

    pill_color = (
        (0, 255, 0)
        if pill_detected
        else (0, 165, 255)
    )

    cv2.putText(
        output,
        pill_tray_status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        pill_status,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        pill_color,
        2
    )

    cv2.putText(
        output,
        f"PILLS: {pill_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        pill_color,
        2
    )

    if pill_detected:
        cv2.putText(
            output,
            f"AREA: {pill_area:.1f}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

    return output



# ============================================================
# CALCULATE MOUTH RATIO
#
# MediaPipe Face Mesh landmark indices:
#
# 61  = left mouth corner
# 291 = right mouth corner
#
# 13  = upper inner lip
# 14  = lower inner lip
# ============================================================

def calculate_mouth_ratio(landmarks):

    try:

        left_corner = landmarks[61]
        right_corner = landmarks[291]

        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        mouth_width = np.sqrt(

            (right_corner.x - left_corner.x) ** 2 +

            (right_corner.y - left_corner.y) ** 2

        )

        mouth_height = np.sqrt(

            (lower_lip.x - upper_lip.x) ** 2 +

            (lower_lip.y - upper_lip.y) ** 2

        )

        if mouth_width <= 0:

            return 0.0

        return mouth_height / mouth_width

    except Exception:

        return 0.0


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


    hand_count = len(
        result.hand_landmarks
    )


    for hand_index, hand_landmarks in enumerate(

        result.hand_landmarks

    ):


        # ----------------------------------------------------
        # Convert normalized landmarks to pixel coordinates
        # ----------------------------------------------------

        points = []


        for landmark in hand_landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append(
                (x, y)
            )


        # ----------------------------------------------------
        # Draw hand connections
        # ----------------------------------------------------

        for start, end in HAND_CONNECTIONS:

            if (

                start < len(points)

                and

                end < len(points)

            ):

                cv2.line(

                    frame,

                    points[start],

                    points[end],

                    (0, 255, 0),

                    2

                )


        # ----------------------------------------------------
        # Draw hand landmarks
        # ----------------------------------------------------

        for point in points:

            cv2.circle(

                frame,

                point,

                4,

                (0, 0, 255),

                -1

            )


        # ----------------------------------------------------
        # Get handedness
        # ----------------------------------------------------

        hand_label = "Hand"

        confidence = 0.0


        if hand_index < len(result.handedness):

            category = (

                result.handedness[hand_index][0]

            )

            hand_label = (

                category.category_name

            )

            confidence = (

                category.score

            )


        # ----------------------------------------------------
        # Display hand label
        # ----------------------------------------------------

        if len(points) > 0:

            wrist_x, wrist_y = points[0]


            text = (

                f"{hand_label} "

                f"{confidence:.2f}"

            )


            cv2.putText(

                frame,

                text,

                (

                    wrist_x,

                    max(

                        30,

                        wrist_y - 15

                    )

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2

            )


    return frame

# ============================================================
# PROCESS PATIENT FRAME
#
# Detect:
#
# 1. Face
# 2. Mouth
# 3. Hand
# ============================================================

def process_patient_frame(frame):

    global detection_status
    global hand_timestamp_ms
    global recognition_frame_counter

    if frame is None:

        return None

    # --------------------------------------------------------
    # STATE-CONTROLLED AI
    #
    # Do not run any patient AI while the medication state does
    # not require it. The camera remains live; only inference is
    # disabled.
    # --------------------------------------------------------
    run_face_ai = patient_face_ai_should_run()
    run_hand_ai = patient_hand_ai_should_run()
    run_recognition = patient_recognition_should_run()

    rgb_frame = frame.copy()

    if not run_face_ai and not run_hand_ai:

        detection_status = {
            "face_count": 0,
            "face_detected": False,
            "mouth_open": False,
            "mouth_ratio": 0.0,
            "hand_detected": False,
            "hand_count": 0,
            "recognized_patient_id": None,
            "recognized_patient_name": None,
            "face_match_score": 0.0,
            "face_recognition_status": "AI STANDBY",
        }

        cv2.putText(
            rgb_frame,
            "AI STANDBY",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            rgb_frame,
            f"STATE: {medication_state}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        return rgb_frame


    # Picamera RGB888 output
    height, width = rgb_frame.shape[:2]


    # --------------------------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame,
    )


    # --------------------------------------------------------
    # DEFAULT STATUS
    # --------------------------------------------------------

    face_count = 0
    mouth_open = False
    mouth_ratio = 0.0

    hand_count = 0

    # --------------------------------------------------------
    # FACE IDENTITY RECOGNITION
    # --------------------------------------------------------
    # MediaPipe remains responsible for face/mouth landmarks.
    # YuNet + SFace independently identifies the detected face.
    # Recognition is throttled to reduce Raspberry Pi CPU load.
    if run_recognition:
        recognition_frame_counter += 1
    else:
        recognition_frame_counter = 0

    if run_recognition and recognition_frame_counter >= RECOGNITION_EVERY_N_FRAMES:
        recognition_frame_counter = 0
        try:
            bgr_for_recognition = cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGB2BGR,
            )
            recognize_patient(bgr_for_recognition)
        except Exception as error:
            print("Face recognition error:", error)

    # --------------------------------------------------------
    # FACE + MOUTH DETECTION
    # --------------------------------------------------------

    if run_face_ai:
        try:

            with mediapipe_lock:

                face_result = face_landmarker.detect(
                    mp_image
                )


            if face_result.face_landmarks:

                face_count = len(
                    face_result.face_landmarks
                )


                # Process every detected face

                for face_landmarks in face_result.face_landmarks:

                    xs = []
                    ys = []

                    for landmark in face_landmarks:

                        x = int(
                            landmark.x * width
                        )

                        y = int(
                            landmark.y * height
                        )

                        xs.append(x)
                        ys.append(y)


                    # Bounding box

                    x1 = max(0, min(xs))
                    y1 = max(0, min(ys))

                    x2 = min(width - 1, max(xs))
                    y2 = min(height - 1, max(ys))


                    # Draw face rectangle

                    cv2.rectangle(

                        rgb_frame,

                        (x1, y1),

                        (x2, y2),

                        (0, 255, 0),

                        2,

                    )


                    # Mouth landmarks

                    mouth_indices = [

                        61,
                        291,
                        13,
                        14,

                    ]


                    for index in mouth_indices:

                        landmark = face_landmarks[index]

                        x = int(
                            landmark.x * width
                        )

                        y = int(
                            landmark.y * height
                        )

                        cv2.circle(

                            rgb_frame,

                            (x, y),

                            4,

                            (255, 0, 0),

                            -1,

                        )


                    # Calculate mouth ratio

                    current_ratio = calculate_mouth_ratio(
                        face_landmarks
                    )


                    # Keep largest ratio

                    if current_ratio > mouth_ratio:

                        mouth_ratio = current_ratio


                # ------------------------------------------------
                # MOUTH OPEN THRESHOLD
                #
                # Can be adjusted after testing
                # ------------------------------------------------

                MOUTH_OPEN_THRESHOLD = 0.22


                if mouth_ratio >= MOUTH_OPEN_THRESHOLD:

                    mouth_open = True


        except Exception as error:

            print("Face detection error:", error)


    # --------------------------------------------------------
    # HAND DETECTION
    # --------------------------------------------------------

    if run_hand_ai:
        try:

            # ----------------------------------------------------
            # Generate strictly increasing timestamp
            #
            # Camera is configured for 20 FPS.
            # 50 ms per frame.
            # ----------------------------------------------------

            hand_timestamp_ms += 50


            # ----------------------------------------------------
            # Run MediaPipe hand detection in VIDEO mode
            # ----------------------------------------------------

            with mediapipe_lock:

                hand_result = hand_landmarker.detect_for_video(

                    mp_image,

                    hand_timestamp_ms

                )


            # ----------------------------------------------------
            # Count detected hands
            # ----------------------------------------------------

            if hand_result.hand_landmarks:

                hand_count = len(
                    hand_result.hand_landmarks
                )

            else:

                hand_count = 0


            # ----------------------------------------------------
            # Draw complete hand skeleton
            # ----------------------------------------------------

            rgb_frame = draw_hand_landmarks(

                rgb_frame,

                hand_result

            )


        except Exception as error:

            print(
                "Hand detection error:",
                error
            )


    # --------------------------------------------------------
    # DRAW FACE IDENTITY
    # --------------------------------------------------------

    if recognized_face_status == "RECOGNIZED":
        identity_text = (
            f"{recognized_patient_name} "
            f"({recognized_patient_id}) "
            f"{recognized_face_score:.2f}"
        )
        identity_color = (0, 255, 0)

    elif recognized_face_status == "UNKNOWN":
        identity_text = "UNKNOWN PATIENT"
        identity_color = (0, 165, 255)

    elif recognized_face_status == "MULTIPLE FACES":
        identity_text = "MULTIPLE FACES"
        identity_color = (0, 0, 255)

    elif recognized_face_status == "NO FACE":
        identity_text = "PATIENT: --"
        identity_color = (255, 0, 0)

    else:
        identity_text = recognized_face_status
        identity_color = (0, 165, 255)

    cv2.putText(
        rgb_frame,
        identity_text,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        identity_color,
        2,
    )

    # --------------------------------------------------------
    # UPDATE GLOBAL STATUS
    # --------------------------------------------------------

    detection_status = {

        "face_count": face_count,

        "face_detected":
            face_count == 1,

        "mouth_open":
            mouth_open,

        "mouth_ratio":
            round(
                float(mouth_ratio),
                3,
            ),

        "hand_detected":
            hand_count > 0,

        "hand_count":
            hand_count,

        "recognized_patient_id":
            recognized_patient_id,

        "recognized_patient_name":
            recognized_patient_name,

        "face_match_score":
            round(float(recognized_face_score), 4),

        "face_recognition_status":
            recognized_face_status,

    }


    # --------------------------------------------------------
    # STATUS TEXT
    # --------------------------------------------------------

    if face_count == 0:

        face_text = "NO FACE"

        face_color = (255, 0, 0)


    elif face_count == 1:

        face_text = "FACE DETECTED"

        face_color = (0, 255, 0)


    else:

        face_text = f"{face_count} FACES"

        face_color = (255, 0, 0)


    mouth_text = (

        "MOUTH OPEN"

        if mouth_open

        else "MOUTH CLOSED"

    )


    mouth_color = (

        (0, 255, 0)

        if mouth_open

        else (255, 165, 0)

    )


    hand_text = (

        f"HANDS: {hand_count}"

    )


    # --------------------------------------------------------
    # DRAW STATUS
    # --------------------------------------------------------

    cv2.putText(

        rgb_frame,

        face_text,

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        face_color,

        2,

    )


    cv2.putText(

        rgb_frame,

        mouth_text,

        (20, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        mouth_color,

        2,

    )


    cv2.putText(

        rgb_frame,

        hand_text,

        (20, 95),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 0),

        2,

    )


    return rgb_frame


# ============================================================
# VIDEO STREAM GENERATOR
# ============================================================

def patient_video_generator():

    while True:

        frame = get_patient_frame()


        if frame is None:

            time.sleep(0.1)

            continue


        processed_frame = process_patient_frame(
            frame
        )


        # Convert RGB -> BGR for JPEG encoding

        bgr_frame = cv2.cvtColor(

            processed_frame,

            cv2.COLOR_RGB2BGR,

        )


        success, buffer = cv2.imencode(

            ".jpg",

            bgr_frame,

            [

                cv2.IMWRITE_JPEG_QUALITY,

                JPEG_QUALITY,

            ],

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
# PILL CAMERA VIDEO GENERATOR
# ============================================================

def pill_video_generator():

    while True:

        frame = get_pill_frame()


        if frame is None:

            time.sleep(0.1)

            continue


        # Picamera gives RGB888.
        # Convert to BGR for OpenCV tray/pill detection.

        bgr_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )


        # Run tray + pill detection

        processed_frame = process_pill_frame(
            bgr_frame
        )


        # Encode

        success, buffer = cv2.imencode(
            ".jpg",
            processed_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY,
            ]
        )


        if not success:

            continue


        frame_bytes = buffer.tobytes()


        yield (

            b"--frame\r\n"

            +

            b"Content-Type: image/jpeg\r\n\r\n"

            +

            frame_bytes

            +

            b"\r\n"

        )

# ============================================================
# MOTOR FUNCTIONS
# ============================================================

def stop_all_motors():

    pill1_pwm.ChangeDutyCycle(0)

    pill2_pwm.ChangeDutyCycle(0)

    GPIO.output(PILL1_AIN1, GPIO.LOW)
    GPIO.output(PILL1_AIN2, GPIO.LOW)

    GPIO.output(PILL2_BIN1, GPIO.LOW)
    GPIO.output(PILL2_BIN2, GPIO.LOW)

    GPIO.output(MOTOR_STBY, GPIO.LOW)


# ============================================================
# DISPENSE PILL
# ============================================================

def dispense_pill(

    pill_number,

    duration=DEFAULT_DISPENSE_TIME,

    speed=DEFAULT_MOTOR_SPEED,

):

    with motor_lock:

        try:

            duration = float(duration)

            speed = float(speed)

        except Exception:

            duration = DEFAULT_DISPENSE_TIME

            speed = DEFAULT_MOTOR_SPEED


        # Safety limits

        duration = max(
            0.1,
            min(duration, 10.0),
        )

        speed = max(
            0,
            min(speed, 100),
        )


        print(

            f"Dispensing Pill {pill_number} | "
            f"Duration: {duration}s | "
            f"Speed: {speed}%"

        )


        # Enable TB6612

        GPIO.output(
            MOTOR_STBY,
            GPIO.HIGH,
        )


        # ----------------------------------------------------
        # PILL 1
        # ----------------------------------------------------

        if pill_number == 1:

            GPIO.output(
                PILL1_AIN1,
                GPIO.HIGH,
            )

            GPIO.output(
                PILL1_AIN2,
                GPIO.LOW,
            )

            pill1_pwm.ChangeDutyCycle(
                speed
            )


            time.sleep(
                duration
            )


            pill1_pwm.ChangeDutyCycle(
                0
            )

            GPIO.output(
                PILL1_AIN1,
                GPIO.LOW,
            )

            GPIO.output(
                PILL1_AIN2,
                GPIO.LOW,
            )


        # ----------------------------------------------------
        # PILL 2
        # ----------------------------------------------------

        elif pill_number == 2:

            GPIO.output(
                PILL2_BIN1,
                GPIO.HIGH,
            )

            GPIO.output(
                PILL2_BIN2,
                GPIO.LOW,
            )

            pill2_pwm.ChangeDutyCycle(
                speed
            )


            time.sleep(
                duration
            )


            pill2_pwm.ChangeDutyCycle(
                0
            )

            GPIO.output(
                PILL2_BIN1,
                GPIO.LOW,
            )

            GPIO.output(
                PILL2_BIN2,
                GPIO.LOW,
            )


        # Disable driver

        GPIO.output(
            MOTOR_STBY,
            GPIO.LOW,
        )


        print(
            f"Pill {pill_number} dispensing complete."
        )


# ============================================================
# START MEDICATION CYCLE
# ============================================================

def start_medication_cycle(
    pill_number,
    patient_id=None,
    schedule_id=None
):

    global medication_state

    global medication_active

    global medication_pill_number

    global medication_start_time

    global medication_state_time

    global medication_pill_count_at_verification

    global medication_patient_id

    global medication_schedule_id


    with medication_lock:


        # ----------------------------------------------------
        # Do not allow two medication cycles simultaneously
        # ----------------------------------------------------

        if medication_active:

            return (
                False,
                "Medication cycle already running."
            )


        # ----------------------------------------------------
        # Validate pill number
        # ----------------------------------------------------

        if pill_number not in [
            1,
            2
        ]:

            return (
                False,
                "Invalid pill number."
            )


        # ----------------------------------------------------
        # Initialize medication cycle
        # ----------------------------------------------------

        medication_active = True

        medication_pill_number = pill_number

        medication_patient_id = patient_id

        medication_schedule_id = schedule_id

        medication_start_time = (
            time.time()
        )

        medication_state_time = (
            time.time()
        )

        medication_pill_count_at_verification = 0

        medication_state = (
            MEDICATION_WAITING_FOR_PATIENT
        )


    add_medication_log(
        event="MEDICATION_CYCLE_STARTED",
        result="STARTED",
        message=(
            f"Medication cycle started "
            f"for pill {pill_number}"
        )
    )


    print("")

    print(
        "=" * 50
    )

    print(
        "MEDICATION CYCLE STARTED"
    )

    print(
        f"PILL NUMBER: {pill_number}"
    )

    print(
        f"PATIENT ID: {patient_id}"
    )

    print(
        f"SCHEDULE ID: {schedule_id}"
    )

    print(
        "=" * 50
    )


    # --------------------------------------------------------
    # Start state machine
    # --------------------------------------------------------

    thread = threading.Thread(
        target=medication_state_machine,
        daemon=True
    )

    thread.start()


    return (
        True,
        "Medication cycle started."
    )


# ============================================================
# MEDICATION STATE MACHINE
# ============================================================

def medication_state_machine():
    """Run the medication state machine behind a fail-safe wrapper."""

    global medication_active
    global medication_pill_number
    global medication_patient_id
    global medication_schedule_id
    global medication_state

    try:
        _medication_state_machine_impl()

    except Exception as error:
        # Never leave a failed medication worker marked active.
        print(
            "MEDICATION STATE MACHINE ERROR:",
            error
        )

        try:
            add_medication_log(
                event="MEDICATION_SYSTEM_ERROR",
                result="FAILED",
                message=(
                    f"Medication state machine stopped unexpectedly: {error}"
                )
            )
        except Exception as log_error:
            print(
                "Medication error log failed:",
                log_error
            )

        try:
            stop_all_motors()
        except Exception as motor_error:
            print(
                "Emergency motor stop failed:",
                motor_error
            )

        medication_active = False
        medication_pill_number = None
        medication_patient_id = None
        medication_schedule_id = None
        medication_state = MEDICATION_IDLE


def _medication_state_machine_impl():

    global medication_state

    global medication_patient_id

    global medication_active

    global medication_pill_number

    global medication_state_time
    global medication_pill_count_at_verification

    last_debug_state = None
    last_patient_debug = None

    # ========================================================
    # WAIT FOR PATIENT
    # ========================================================

    print(
        "STATE: WAITING FOR PATIENT"
    )


    while medication_active:


        if medication_state != last_debug_state:

            print("")

            print("========================================")

            print(
                f"MEDICATION STATE: {medication_state}"
            )

            print(
                f"PILL NUMBER: {medication_pill_number}"
            )

            print("========================================")

            last_debug_state = medication_state



        # ----------------------------------------------------
        # STATE:
        #
        # WAITING FOR PATIENT
        # ----------------------------------------------------

        if (

            medication_state ==

            MEDICATION_WAITING_FOR_PATIENT

        ):


            # --------------------------------------------
            # Recognized patient detected
            # --------------------------------------------

            detected_patient_id = detection_status.get(
                "recognized_patient_id"
            )

            scheduled_patient_id = (
                str(medication_patient_id)
                if medication_patient_id is not None
                else None
            )

            patient_match = (
                detection_status.get("face_detected", False)
                and detected_patient_id is not None
                and (
                    scheduled_patient_id is None
                    or detected_patient_id == scheduled_patient_id
                )
            )

            # Show the actual live recognition decision when it changes.
            patient_debug = (
                detection_status.get("face_recognition_status"),
                detected_patient_id,
                scheduled_patient_id,
                patient_match,
            )

            if patient_debug != last_patient_debug:
                print(
                    "PATIENT CHECK | "
                    f"recognition={patient_debug[0]} | "
                    f"recognized={detected_patient_id or 'NONE'} | "
                    f"scheduled={scheduled_patient_id or 'ANY'} | "
                    f"match={'YES' if patient_match else 'NO'}"
                )
                last_patient_debug = patient_debug

            if patient_match:

                print(
                    f"PATIENT RECOGNIZED: {detected_patient_id}"
                )

                add_medication_log(
                    event="PATIENT_RECOGNIZED",
                    result="SUCCESS",
                    message=(
                        f"Recognized patient "
                        f"{detected_patient_id}."
                    )
                )

                medication_state = (

                    MEDICATION_DISPENSING

                )

                medication_state_time = (

                    time.time()

                )

                continue

            # A face can be present but it must be a registered
            # patient, and for a scheduled cycle it must be the
            # patient assigned to that schedule.


            # --------------------------------------------
            # Timeout
            # --------------------------------------------

            elapsed = (

                time.time()

                -

                medication_state_time

            )


            if elapsed > PATIENT_TIMEOUT:

                medication_state = (

                    MEDICATION_TIMEOUT

                )

                print(
                    "PATIENT DETECTION TIMEOUT"
                )

                continue


        # ----------------------------------------------------
        # DISPENSING
        # ----------------------------------------------------

        elif (

            medication_state ==

            MEDICATION_DISPENSING

        ):


            print(

                f"DISPENSING PILL "

                f"{medication_pill_number}"

            )

            add_medication_log(
                event="PILL_DISPENSING",
                result="STARTED",
                message=(
                    f"Dispensing pill "
                    f"{medication_pill_number}"
                )
            )


            try:

                # ----------------------------------------
                # Use your existing motor function
                # ----------------------------------------

                dispense_pill(

                    pill_number=medication_pill_number,

                    duration=DEFAULT_DISPENSE_TIME,

                    speed=DEFAULT_MOTOR_SPEED
                )


                print(
                    "PILL DISPENSE COMMAND COMPLETE"
                )


                add_medication_log(
                    event="PILL_DISPENSED",
                    result="SUCCESS",
                    message="Motor dispensing completed."
                )

                medication_state = (

                    MEDICATION_WAITING_FOR_PILL

                )

                medication_state_time = (

                    time.time()

                )


            except Exception as error:

                print(

                    "Dispensing error:",

                    error

                )


                medication_state = (

                    MEDICATION_TIMEOUT

                )


            continue


        # ----------------------------------------------------
        # WAIT FOR PILL TO APPEAR
        # ----------------------------------------------------

        elif (

            medication_state ==

            MEDICATION_WAITING_FOR_PILL

        ):


            if pill_detected:

                medication_pill_count_at_verification = max(
                    1,
                    int(pill_count)
                )

                pill_removal_counter = 0

                print(
                    f"PILL DETECTED IN TRAY | COUNT: {pill_count}"
                )

                add_medication_log(
                    event="PILL_VERIFIED",
                    result="SUCCESS",
                    message="Pill detected in dispensing tray."
                )

                medication_state = (

                    MEDICATION_WAITING_FOR_PICKUP

                )

                medication_state_time = (

                    time.time()

                )

                continue


            elapsed = (

                time.time()

                -

                medication_state_time

            )


            if elapsed > PILL_DETECTION_TIMEOUT:

                print(
                    "PILL DETECTION TIMEOUT"
                )


                medication_state = (

                    MEDICATION_TIMEOUT

                )

                continue


        # ----------------------------------------------------
        # WAIT FOR PILL TO BE PICKED
        # ----------------------------------------------------

        elif (

            medication_state ==

            MEDICATION_WAITING_FOR_PICKUP

        ):


            # --------------------------------------------
            # Pill was present.
            #
            # Now it disappears from the tray.
            # --------------------------------------------

            pill_count_decreased = (
                pill_count < medication_pill_count_at_verification
            )

            if pill_count_decreased or not pill_detected:
                pill_removal_counter += 1
            else:
                pill_removal_counter = 0

            if pill_removal_counter >= REQUIRED_REMOVAL_FRAMES:

                print(
                    "PILL REMOVED FROM TRAY"
                )

                add_medication_log(
                    event="PILL_PICKED_UP",
                    result="SUCCESS",
                    message="Pill removed from tray."
                )


                medication_state = (

                    MEDICATION_WAITING_FOR_HAND

                )

                medication_state_time = (

                    time.time()

                )

                pill_removal_counter = 0

                continue


            elapsed = (

                time.time()

                -

                medication_state_time

            )


            if elapsed > PICKUP_TIMEOUT:

                print(
                    "PILL PICKUP TIMEOUT"
                )


                medication_state = (

                    MEDICATION_TIMEOUT

                )

                continue


        # ----------------------------------------------------
        # WAIT FOR HAND
        # ----------------------------------------------------

        elif (

            medication_state ==

            MEDICATION_WAITING_FOR_HAND

        ):


            if detection_status.get(

                "hand_detected",

                False

            ):

                print(
                    "HAND DETECTED"
                )

                add_medication_log(
                    event="HAND_DETECTED",
                    result="SUCCESS",
                    message="Hand detected after pill pickup."
                )

                medication_state = (

                    MEDICATION_WAITING_FOR_MOUTH

                )

                medication_state_time = (

                    time.time()

                )

                continue


            elapsed = (

                time.time()

                -

                medication_state_time

            )


            if elapsed > HAND_TIMEOUT:

                print(
                    "HAND DETECTION TIMEOUT"
                )


                medication_state = (

                    MEDICATION_TIMEOUT

                )

                continue


        # ----------------------------------------------------
        # WAIT FOR MOUTH OPEN
        # ----------------------------------------------------

        elif (

            medication_state ==

            MEDICATION_WAITING_FOR_MOUTH

        ):


            # --------------------------------------------
            # We require:
            #
            # 1. Face detected
            # 2. Hand detected
            # 3. Mouth open
            # --------------------------------------------

            face_present = (

                detection_status.get(

                    "face_detected",

                    False

                )

            )


            hand_present = (

                detection_status.get(

                    "hand_detected",

                    False

                )

            )


            mouth_open = (

                detection_status.get(

                    "mouth_open",

                    False

                )

            )


            if (

                face_present

                and

                hand_present

                and

                mouth_open

            ):


                print("")

                print(

                    "========================================"

                )

                print(
                    "MEDICATION TAKEN"
                )

                print(

                    "========================================"

                )


                add_medication_log(
                    event="MEDICATION_TAKEN",
                    result="SUCCESS",
                    message="Face, hand and mouth conditions satisfied."
                )


                start_buzzer_pattern(
                    count=2,
                    on_time=0.1,
                    off_time=0.1
                )

                medication_state = (

                    MEDICATION_TAKEN

                )

                medication_state_time = (

                    time.time()

                )

                continue


            elapsed = (

                time.time()

                -

                medication_state_time

            )


            if elapsed > MOUTH_TIMEOUT:

                print(
                    "MOUTH DETECTION TIMEOUT"
                )


                medication_state = (

                    MEDICATION_TIMEOUT

                )

                continue


        # ----------------------------------------------------
        # MEDICATION TAKEN
        # ----------------------------------------------------

        elif (

            medication_state ==

            MEDICATION_TAKEN

        ):


            # --------------------------------------------
            # Keep success state visible briefly
            # --------------------------------------------

            time.sleep(3)


            medication_active = False


            medication_pill_number = None


            medication_state = (

                MEDICATION_IDLE

            )


            print(
                "MEDICATION CYCLE COMPLETE"
            )


            break


        # ----------------------------------------------------
        # TIMEOUT / ERROR
        # ----------------------------------------------------

        elif (

            medication_state ==

            MEDICATION_TIMEOUT

        ):


            print(
                "MEDICATION CYCLE FAILED / TIMED OUT"
            )

            add_medication_log(
                event="MEDICATION_TIMEOUT",
                result="FAILED",
                message="Medication cycle timed out."
            )


            start_buzzer_pattern(
                count=5,
                on_time=0.1,
                off_time=0.1
            )

            time.sleep(3)


            medication_active = False

            medication_pill_number = None

            medication_patient_id = None

            medication_schedule_id = None

            medication_state = (

                MEDICATION_IDLE

            )


            break


        # ----------------------------------------------------
        # Small delay
        # ----------------------------------------------------

        time.sleep(

            0.05

        )

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )

# ============================================================
# START MEDICATION CYCLE
# ============================================================

@app.route(

    "/api/start_medication",

    methods=["POST"]

)

def api_start_medication():

    try:


        data = request.get_json(

            silent=True

        )


        if data is None:

            data = {}


        pill_number = data.get(

            "pill_number",

            1

        )


        pill_number = int(

            pill_number

        )


        success, message = (

            start_medication_cycle(

                pill_number

            )

        )


        return jsonify(

            {

                "success": success,

                "message": message,

                "state": medication_state,

            }

        )


    except Exception as error:


        print(

            "Medication start error:",

            error

        )


        return jsonify(

            {

                "success": False,

                "message": str(error),

            }

        ), 500


# ============================================================
# CAPTURE EMPTY TRAY REFERENCE
# ============================================================

@app.route(
    "/api/capture_empty_tray",
    methods=["POST"]
)
def api_capture_empty_tray():

    try:

        success, message = (
            capture_empty_tray_reference()
        )


        return jsonify(
            {
                "success": success,
                "message": message,
            }
        )


    except Exception as error:

        print(
            "Empty tray capture error:",
            error
        )


        return jsonify(
            {
                "success": False,
                "message": str(error),
            }
        ), 500

# ============================================================
# PATIENT CAMERA STREAM
# ============================================================

@app.route("/patient_video_feed")
def patient_video_feed():

    return Response(

        patient_video_generator(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),

    )


# ============================================================
# PILL CAMERA STREAM
# ============================================================

@app.route("/pill_video_feed")
def pill_video_feed():

    return Response(

        pill_video_generator(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),

    )


# ============================================================
# GET DETECTION STATUS
# ============================================================

@app.route("/api/status")
def get_status():

    status = detection_status.copy()


    status.update(
        {

            "tray_detected":
                pill_tray_detected,

            "tray_status":
                pill_tray_status,

            "pill_detected":
                pill_detected,

            "pill_status":
                pill_status,

            "pill_area":
                round(
                    float(pill_area),
                    1
                ),

            "pill_bbox":
                pill_bbox,

            "pill_count":
                pill_count,

            "pill_confidences":
                pill_confidences,

            "yolo_pill_detector_loaded":
                pill_yolo_net is not None,

            "empty_reference_loaded":
                reference_image is not None,

            "medication_state":
                medication_state,

            "medication_active":
                medication_active,

            "medication_pill_number":
                medication_pill_number,

            "medication_patient_id":
                medication_patient_id,

            "scheduled_patient_id":
                medication_patient_id,

            "recognized_patient_id":
                recognized_patient_id,

            "recognized_patient_name":
                recognized_patient_name,

            "face_match_score":
                round(float(recognized_face_score), 4),

            "face_recognition_status":
                recognized_face_status,

            "medication_schedule_id":
                medication_schedule_id,

            "current_time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "scheduler_running":
                scheduler_running,

            "recognized_patient_id":
                recognized_patient_id,

            "recognized_patient_name":
                recognized_patient_name,

            "face_match_score":
                round(float(recognized_face_score), 4),

            "face_recognition_status":
                recognized_face_status,

        }
    )


    return jsonify(
        status
    )

# ============================================================
# GET MEDICATION SCHEDULE
# ============================================================

@app.route(
    "/api/medication_schedule"
)
def get_medication_schedule():

    schedules = (
        load_medication_schedule()
    )

    return jsonify(
        schedules
    )


# ============================================================
# ADD MEDICATION SCHEDULE
# ============================================================

@app.route(
    "/api/medication_schedule",
    methods=["POST"]
)
def add_medication_schedule():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        schedule_id = str(
            data.get(
                "schedule_id",
                ""
            )
        ).strip()


        patient_id = data.get(
            "patient_id",
            None
        )


        pill_number = int(
            data.get(
                "pill_number",
                0
            )
        )


        schedule_time = str(
            data.get(
                "time",
                ""
            )
        ).strip()


        enabled = bool(
            data.get(
                "enabled",
                True
            )
        )


        if not schedule_id:

            return jsonify(
                {
                    "success": False,
                    "message":
                        "schedule_id is required."
                }
            ), 400


        if pill_number not in [
            1,
            2
        ]:

            return jsonify(
                {
                    "success": False,
                    "message":
                        "Invalid pill number."
                }
            ), 400


        try:

            datetime.strptime(
                schedule_time,
                "%H:%M"
            )

        except ValueError:

            return jsonify(
                {
                    "success": False,
                    "message":
                        "Time must be HH:MM."
                }
            ), 400


        schedules = (
            load_medication_schedule()
        )


        for schedule in schedules:

            if schedule.get(
                "schedule_id"
            ) == schedule_id:

                return jsonify(
                    {
                        "success": False,
                        "message":
                            "Schedule ID already exists."
                    }
                ), 400


        new_schedule = {

            "schedule_id":
                schedule_id,

            "patient_id":
                patient_id,

            "pill_number":
                pill_number,

            "time":
                schedule_time,

            "enabled":
                enabled,

            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }


        schedules.append(
            new_schedule
        )


        if not save_medication_schedule(
            schedules
        ):

            return jsonify(
                {
                    "success": False,
                    "message":
                        "Could not save schedule."
                }
            ), 500


        return jsonify(
            {
                "success": True,
                "message":
                    "Medication schedule added.",
                "schedule":
                    new_schedule
            }
        )


    except Exception as error:

        print(
            "Schedule creation error:",
            error
        )

        return jsonify(
            {
                "success": False,
                "message":
                    str(error)
            }
        ), 500


# ============================================================
# UPDATE MEDICATION SCHEDULE
# ============================================================

@app.route(
    "/api/medication_schedule/<schedule_id>",
    methods=["PUT"]
)
def update_medication_schedule(
    schedule_id
):

    try:

        data = request.get_json(
            silent=True
        ) or {}

        schedules = load_medication_schedule()

        target = None

        for schedule in schedules:
            if str(schedule.get("schedule_id")) == str(schedule_id):
                target = schedule
                break

        if target is None:
            return jsonify(
                {
                    "success": False,
                    "message": "Schedule not found."
                }
            ), 404

        if "time" in data:
            schedule_time = str(data.get("time", "")).strip()
            try:
                datetime.strptime(
                    schedule_time,
                    "%H:%M"
                )
            except ValueError:
                return jsonify(
                    {
                        "success": False,
                        "message": "Time must be HH:MM."
                    }
                ), 400
            target["time"] = schedule_time

        if "pill_number" in data:
            try:
                pill_number = int(data.get("pill_number"))
            except (TypeError, ValueError):
                return jsonify(
                    {
                        "success": False,
                        "message": "Invalid pill number."
                    }
                ), 400

            if pill_number not in [1, 2]:
                return jsonify(
                    {
                        "success": False,
                        "message": "Invalid pill number."
                    }
                ), 400

            target["pill_number"] = pill_number

        if "medicine_name" in data:
            target["medicine_name"] = str(
                data.get("medicine_name", "")
            ).strip()

        if "quantity" in data:
            try:
                quantity = int(data.get("quantity"))
            except (TypeError, ValueError):
                return jsonify(
                    {
                        "success": False,
                        "message": "Quantity must be a positive number."
                    }
                ), 400

            if quantity < 1:
                return jsonify(
                    {
                        "success": False,
                        "message": "Quantity must be a positive number."
                    }
                ), 400

            target["quantity"] = quantity

        if "enabled" in data:
            target["enabled"] = bool(data.get("enabled"))

        target["updated_at"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        if not save_medication_schedule(schedules):
            return jsonify(
                {
                    "success": False,
                    "message": "Could not save schedule changes."
                }
            ), 500

        return jsonify(
            {
                "success": True,
                "message": "Medication schedule updated.",
                "schedule": target
            }
        )

    except Exception as error:

        print(
            "Schedule update error:",
            error
        )

        return jsonify(
            {
                "success": False,
                "message": str(error)
            }
        ), 500


# ============================================================
# DELETE MEDICATION SCHEDULE
# ============================================================

@app.route(
    "/api/medication_schedule/<schedule_id>",
    methods=["DELETE"]
)
def delete_medication_schedule(
    schedule_id
):

    schedules = (
        load_medication_schedule()
    )


    updated_schedules = [

        schedule

        for schedule in schedules

        if str(
            schedule.get(
                "schedule_id"
            )
        )
        !=
        str(
            schedule_id
        )
    ]


    if len(
        updated_schedules
    ) == len(
        schedules
    ):

        return jsonify(
            {
                "success": False,
                "message":
                    "Schedule not found."
            }
        ), 404


    if not save_medication_schedule(
        updated_schedules
    ):

        return jsonify(
            {
                "success": False,
                "message":
                    "Could not update schedule."
            }
        ), 500


    return jsonify(
        {
            "success": True,
            "message":
                "Schedule deleted."
        }
    )

# ============================================================
# GET MEDICATION LOGS
# ============================================================

@app.route(
    "/api/medication_logs"
)
def get_medication_logs():

    logs = load_medication_logs()

    limit = request.args.get(
        "limit",
        default=50,
        type=int
    )


    limit = max(
        1,
        min(
            limit,
            1000
        )
    )


    logs = logs[-limit:]


    logs.reverse()


    return jsonify(
        logs
    )

# ============================================================
# GET ALL PATIENTS
# ============================================================

@app.route("/api/patients")
def get_patients():

    patients = load_patients()

    public_patients = []

    for patient in patients:
        public_patient = dict(patient)
        public_patient.pop("face_embedding", None)
        public_patients.append(public_patient)

    return jsonify(
        public_patients
    )


# ============================================================
# REGISTER PATIENT
#
# Expected JSON:
#
# {
#     "patient_id": "...",
#     "name": "...",
#     "image": "data:image/jpeg;base64,..."
# }
# ============================================================

@app.route(
    "/api/register_patient",
    methods=["POST"],
)
def register_patient():

    try:

        data = request.get_json()


        if not data:

            return jsonify(
                {
                    "success": False,
                    "message": "No patient data received.",
                }
            ), 400


        patient_id = str(
            data.get(
                "patient_id",
                "",
            )
        ).strip()


        name = str(
            data.get(
                "name",
                "",
            )
        ).strip()


        image_data = data.get(
            "image",
            "",
        )


        if not patient_id:

            return jsonify(
                {
                    "success": False,
                    "message": "Patient ID is required.",
                }
            ), 400


        if not name:

            return jsonify(
                {
                    "success": False,
                    "message": "Patient name is required.",
                }
            ), 400


        if not image_data:

            return jsonify(
                {
                    "success": False,
                    "message": "Patient face image is required.",
                }
            ), 400


        # ----------------------------------------------------
        # REMOVE DATA URL HEADER
        # ----------------------------------------------------

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1,
            )[1]


        patients = load_patients()


        # ----------------------------------------------------
        # CHECK DUPLICATE PATIENT ID
        # ----------------------------------------------------

        for patient in patients:

            if patient.get(
                "patient_id"
            ) == patient_id:

                return jsonify(
                    {
                        "success": False,
                        "message":
                            "Patient ID already exists.",
                    }
                ), 400


        # ----------------------------------------------------
        # GENERATE FACE EMBEDDING
        # ----------------------------------------------------

        registration_image = decode_patient_image(
            image_data
        )

        if registration_image is None:
            return jsonify(
                {
                    "success": False,
                    "message":
                        "Could not decode patient face image.",
                }
            ), 400

        face_embedding, embedding_error = (
            create_face_embedding(
                registration_image
            )
        )

        if embedding_error:
            return jsonify(
                {
                    "success": False,
                    "message": embedding_error,
                }
            ), 400

        # ----------------------------------------------------
        # CREATE PATIENT
        # ----------------------------------------------------

        medications = data.get(
            "medications",
            []
        )

        patient = {

            "patient_id":
                patient_id,

            "name":
                name,

            "image":
                image_data,

            "face_embedding":
                face_embedding,

            "medications":
                medications,

            "registered_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

        }


        patients.append(
            patient
        )


        if not save_patients(
            patients
        ):

            return jsonify(
                {
                    "success": False,
                    "message":
                        "Could not save patient.",
                }
            ), 500


        # ----------------------------------------------------
        # CREATE MEDICATION SCHEDULES
        # ----------------------------------------------------
        # Registration stores medication details in the patient
        # record, but the scheduler reads MEDICATION_SCHEDULE_FILE.
        # Create schedule entries here so registration immediately
        # activates the supplied times.
        # Only medications with a valid time are scheduled.
        # ----------------------------------------------------
        created_schedules = []
        schedules = load_medication_schedule()

        for medication in medications:
            medication_time = str(
                medication.get("time", "")
            ).strip()

            if not medication_time:
                continue

            try:
                datetime.strptime(
                    medication_time,
                    "%H:%M"
                )
            except ValueError:
                continue

            try:
                pill_number = int(
                    medication.get("pill_number", 0)
                )
            except (TypeError, ValueError):
                continue

            if pill_number not in [1, 2]:
                continue

            schedule = {
                "schedule_id":
                    f"{patient_id}_{pill_number}_{int(time.time() * 1000)}",
                "patient_id":
                    patient_id,
                "pill_number":
                    pill_number,
                "time":
                    medication_time,
                "medicine_name":
                    str(medication.get("medicine_name", "")).strip(),
                "quantity":
                    medication.get("quantity", 1),
                "enabled":
                    True,
                "created_at":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
            }

            schedules.append(schedule)
            created_schedules.append(schedule)

        if created_schedules and not save_medication_schedule(schedules):
            # Patient registration succeeded, but do not leave the
            # user believing schedules were created when persistence failed.
            # Remove the patient we just added and report the error.
            patients = [
                item
                for item in patients
                if item.get("patient_id") != patient_id
            ]
            save_patients(patients)

            return jsonify(
                {
                    "success": False,
                    "message":
                        "Patient was not registered because the medication schedule could not be saved.",
                }
            ), 500


        print(
            f"Patient registered: "
            f"{name} ({patient_id})"
        )

        print(
            f"Medication schedules created: {len(created_schedules)}"
        )


        return jsonify(
            {
                "success": True,
                "message":
                    "Patient registered successfully.",
                "patient":
                    patient,
                "schedules":
                    created_schedules,
            }
        )


    except Exception as error:

        print(
            "Patient registration error:",
            error,
        )

        return jsonify(
            {
                "success": False,
                "message":
                    str(error),
            }
        ), 500


# ============================================================
# DELETE PATIENT
# ============================================================

@app.route(
    "/api/patient/<patient_id>",
    methods=["DELETE"],
)
def delete_patient(
    patient_id
):

    patients = load_patients()


    updated_patients = [

        patient

        for patient in patients

        if patient.get(
            "patient_id"
        ) != patient_id

    ]


    if len(
        updated_patients
    ) == len(patients):

        return jsonify(
            {
                "success": False,
                "message":
                    "Patient not found.",
            }
        ), 404


    save_patients(
        updated_patients
    )


    return jsonify(
        {
            "success": True,
            "message":
                "Patient deleted.",
        }
    )


# ============================================================
# DISPENSE API
# ============================================================

@app.route(
    "/api/dispense",
    methods=["POST"],
)
def dispense():

    try:

        data = request.get_json() or {}

        pill_number = int(
            data.get(
                "pill_number",
                0,
            )
        )


        duration = data.get(

            "duration",

            DEFAULT_DISPENSE_TIME,

        )


        speed = data.get(

            "speed",

            DEFAULT_MOTOR_SPEED,

        )


        if pill_number not in [1, 2]:

            return jsonify(
                {
                    "success": False,
                    "message":
                        "Invalid pill number.",
                }
            ), 400


        # Run motor in separate thread
        # so Flask does not freeze

        motor_thread = threading.Thread(

            target=dispense_pill,

            args=(
                pill_number,
                duration,
                speed,
            ),

            daemon=True,

        )

        motor_thread.start()


        return jsonify(
            {
                "success": True,

                "message":
                    f"Pill {pill_number} dispensing started.",
            }
        )


    except Exception as error:

        print(
            "Dispense error:",
            error,
        )

        return jsonify(
            {
                "success": False,
                "message":
                    str(error),
            }
        ), 500


# ============================================================
# STOP MOTORS API
# ============================================================

@app.route(
    "/api/stop_motors",
    methods=["POST"],
)
def stop_motors():

    try:

        stop_all_motors()

        return jsonify(
            {
                "success": True,
                "message":
                    "All motors stopped.",
            }
        )

    except Exception as error:

        return jsonify(
            {
                "success": False,
                "message":
                    str(error),
            }
        ), 500


# ============================================================
# MIGRATE PATIENT MEDICATIONS TO SCHEDULE FILE
# ============================================================

def migrate_medication_schedules():

    patients = load_patients()
    schedules = load_medication_schedule()
    changed = False
    existing_keys = set()

    for schedule in schedules:
        existing_keys.add(
            (
                str(schedule.get("patient_id")),
                str(schedule.get("pill_number")),
                str(schedule.get("time")),
            )
        )

    for patient in patients:
        patient_id = str(patient.get("patient_id", "")).strip()

        if not patient_id:
            continue

        for medication in patient.get("medications", []) or []:
            medication_time = str(
                medication.get("time", "")
            ).strip()

            if not medication_time:
                continue

            try:
                datetime.strptime(
                    medication_time,
                    "%H:%M"
                )
            except ValueError:
                continue

            try:
                pill_number = int(
                    medication.get("pill_number", 0)
                )
            except (TypeError, ValueError):
                continue

            if pill_number not in [1, 2]:
                continue

            key = (
                patient_id,
                str(pill_number),
                medication_time,
            )

            if key in existing_keys:
                continue

            schedule = {
                "schedule_id":
                    f"{patient_id}_{pill_number}_{int(time.time() * 1000)}",
                "patient_id":
                    patient_id,
                "pill_number":
                    pill_number,
                "time":
                    medication_time,
                "medicine_name":
                    str(medication.get("medicine_name", "")).strip(),
                "quantity":
                    medication.get("quantity", 1),
                "enabled":
                    True,
                "created_at":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
            }

            schedules.append(schedule)
            existing_keys.add(key)
            changed = True

    if changed:
        if save_medication_schedule(schedules):
            print(
                f"Medication schedule migration completed: {len(schedules)} schedules."
            )
        else:
            print(
                "WARNING: Could not save migrated medication schedules."
            )
    else:
        print("Medication schedule migration: no changes needed.")


# ============================================================
# APPLICATION STARTUP
# ============================================================

if __name__ == "__main__":

    try:

        start_cameras()

        load_pill_yolo_model()

        load_empty_tray_reference()

        initialize_mediapipe()

        initialize_face_recognition()

        migrate_patient_embeddings()

        migrate_medication_schedules()

        start_medication_scheduler()

        print("\n")
        print("=" * 60)
        print("SMART PILL DISPENSER")
        print("=" * 60)

        print("\nOpen in browser:\n")

        print(
            "http://RASPBERRY_PI_IP:5000"
        )

        print("\nPatient Camera:")
        print(
            "/patient_video_feed"
        )

        print("\nPill Camera:")
        print(
            "/pill_video_feed"
        )

        print("\nPress CTRL+C to stop.\n")


        app.run(

            host="0.0.0.0",

            port=5000,

            debug=False,

            threaded=True,

        )


    except KeyboardInterrupt:

        print(
            "\nStopping application..."
        )


    except Exception as error:

        print(
            "\nAPPLICATION ERROR:"
        )

        print(error)


    finally:

        try:

            stop_all_motors()

        except Exception:

            pass


        try:

            pill1_pwm.stop()

            pill2_pwm.stop()

        except Exception:

            pass


        try:

            GPIO.cleanup()

        except Exception:

            pass


        try:

            if patient_camera is not None:

                patient_camera.stop()

        except Exception:

            pass


        try:

            if pill_camera is not None:

                pill_camera.stop()

        except Exception:

            pass
