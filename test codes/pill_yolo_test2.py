import cv2
import ncnn
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models/pill_detector_ncnn"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

CAMERA_WIDTH = 1536
CAMERA_HEIGHT = 864

# Reference tray size used ONLY for scale-independent area/shape validation.
# YOLO itself will use the full-resolution dynamic tray crop.
ANALYSIS_WIDTH = 420
ANALYSIS_HEIGHT = 300

# Dynamic tray detection - copied from app1.py
MIN_TRAY_AREA = 2500
MIN_TRAY_WIDTH = 60
MIN_TRAY_HEIGHT = 40

# Pill validation
MIN_PILL_AREA = 100
MAX_PILL_AREA = 1500
MIN_PILL_WIDTH = 8
MIN_PILL_HEIGHT = 8
MIN_SOLIDITY = 0.55

# Temporal stability: require the same pill area to be present
# in consecutive frames before declaring it detected.
REQUIRED_DETECTION_FRAMES = 2

# Red/colored/dark foreground checks used only as a sanity filter
PILL_DARK_VALUE_MAX = 85
PILL_COLOR_SATURATION_MIN = 45
PILL_COLOR_VALUE_MAX = 220
MIN_PILL_COLOR_RATIO = 0.30

app = Flask(__name__)

# ============================================================
# LOAD NCNN MODEL
# ============================================================

print("Loading NCNN model...")

net = ncnn.Net()
ret = net.load_param(f"{MODEL_DIR}/model.ncnn.param")
if ret != 0:
    raise RuntimeError(f"Failed to load NCNN param: {ret}")

ret = net.load_model(f"{MODEL_DIR}/model.ncnn.bin")
if ret != 0:
    raise RuntimeError(f"Failed to load NCNN model: {ret}")

print("NCNN model loaded successfully.")

# ============================================================
# CAMERA
# ============================================================

camera = Picamera2(1)
camera.configure(
    camera.create_video_configuration(
        main={
            "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
            "format": "RGB888"
        }
    )
)
camera.start()

print("Pill camera opened successfully.")

# ============================================================
# LETTERBOX
# ============================================================

def letterbox(image, new_size=640):
    height, width = image.shape[:2]

    scale = min(
        new_size / width,
        new_size / height
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

# ============================================================
# DYNAMIC TRAY DETECTION
# Same dynamic tray approach used by the test pipeline.
#   1. find the bright dispenser/tray contour
#   2. use the complete detected tray interior
#   3. erode the boundary slightly
# ============================================================

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
# Same normalization as app1.py
# ============================================================

def extract_tray_crops(frame, contour, mask):
    """Return a full-resolution tray crop for YOLO plus a 420x300
    analysis copy used only for scale-independent area validation.
    """
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

    # Full-resolution crop: this is what YOLO sees.
    # Do NOT resize this to 420x300.
    analysis_tray = cv2.resize(
        tray,
        (ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    analysis_mask = cv2.resize(
        tray_mask,
        (ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )

    _, analysis_mask = cv2.threshold(
        analysis_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return tray, analysis_tray, analysis_mask, (x1, y1, x2, y2)

# ============================================================
# FOREGROUND / AREA VALIDATION
# This is NOT the YOLO detector.
# It is a safety filter after YOLO has proposed a pill.
# ============================================================

def estimate_pill_area(tray, tray_mask, bbox):
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

    # Candidate pixels that look different from the white tray:
    #   - dark pixels, OR
    #   - sufficiently saturated/colored pixels
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

# ============================================================
# YOLO + TRAY FILTERING
# ============================================================

def detect_pill(frame):
    """Detect and count all valid pills inside the dynamically detected tray.

    Returns:
        frame, detected, pill_count, best_area
    """

    # --------------------------------------------------------
    # 1. Detect tray dynamically from the full camera frame.
    # --------------------------------------------------------
    tray_contour, tray_mask_full = detect_tray(frame)

    if tray_contour is None:
        cv2.putText(
            frame,
            "TRAY NOT FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        return frame, False, 0, 0.0

    # Draw the complete dynamically detected dispenser contour.
    cv2.drawContours(
        frame,
        [tray_contour],
        -1,
        (255, 255, 0),
        2
    )

    # --------------------------------------------------------
    # 2. Extract the full-resolution tray.
    #    A separate 420x300 copy is kept only for area/shape
    #    validation so our old area limits remain comparable.
    # --------------------------------------------------------
    tray, analysis_tray, analysis_mask, crop_box = extract_tray_crops(
        frame,
        tray_contour,
        tray_mask_full
    )

    if tray is None or analysis_tray is None or analysis_mask is None:
        cv2.putText(
            frame,
            "TRAY EXTRACTION FAILED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        return frame, False, 0, 0.0

    # --------------------------------------------------------
    # 3. Run YOLO on the full-resolution dynamic tray crop, not the whole frame.
    # --------------------------------------------------------
    image, scale, pad_x, pad_y = letterbox(
        tray,
        INPUT_SIZE
    )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mat = ncnn.Mat.from_pixels(
        rgb,
        ncnn.Mat.PixelType.PIXEL_RGB,
        INPUT_SIZE,
        INPUT_SIZE
    )

    mat.substract_mean_normalize(
        [],
        [1.0 / 255.0] * 3
    )

    with net.create_extractor() as ex:
        ret = ex.input("in0", mat)
        if ret != 0:
            return frame, False, 0, 0.0

        ret, out = ex.extract("out0")
        if ret != 0:
            return frame, False, 0, 0.0

    output = np.array(out)

    boxes = []
    scores = []

    for i in range(output.shape[1]):
        score = float(output[4, i])

        if score < CONF_THRESHOLD:
            continue

        x_center = float(output[0, i])
        y_center = float(output[1, i])
        width = float(output[2, i])
        height = float(output[3, i])

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0

        boxes.append([x1, y1, width, height])
        scores.append(score)

    if not boxes:
        cv2.putText(
            frame,
            "NO YOLO CANDIDATE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        return frame, False, 0, 0.0

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices) == 0:
        return frame, False, 0, 0.0

    # --------------------------------------------------------
    # 4. Evaluate every NMS candidate.
    #    Do NOT blindly use indices[0].
    # --------------------------------------------------------
    candidates = []

    for raw_index in indices:
        index = int(raw_index)

        x1, y1, box_width, box_height = boxes[index]
        confidence = scores[index]

        x2 = x1 + box_width
        y2 = y1 + box_height

        # Convert 640x640 letterbox coordinates into the
        # full-resolution dynamic tray crop.
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

        if nw < MIN_PILL_WIDTH or nh < MIN_PILL_HEIGHT:
            continue

        # ----------------------------------------------------
        # Center must lie inside the dynamically detected tray
        # mask. The mask is checked in full-resolution tray
        # coordinates, so the YOLO ROI and validation ROI match.
        # ----------------------------------------------------
        center_x = int((nx1 + nx2) / 2)
        center_y = int((ny1 + ny2) / 2)

        if not (
            0 <= center_x < tray_mask_full.shape[1]
            and 0 <= center_y < tray_mask_full.shape[0]
        ):
            continue

        frame_center_x = crop_box[0] + center_x
        frame_center_y = crop_box[1] + center_y

        if not (
            0 <= frame_center_x < tray_mask_full.shape[1]
            and 0 <= frame_center_y < tray_mask_full.shape[0]
            and tray_mask_full[frame_center_y, frame_center_x] > 0
        ):
            continue

        # ----------------------------------------------------
        # Estimate the actual foreground contour area inside
        # the YOLO box. This is the safety size check.
        # The YOLO bounding-box area itself is NOT used because
        # the box can be much larger than the physical pill.
        # ----------------------------------------------------
        # Map the full-resolution YOLO box to the fixed 420x300
        # analysis copy so the 100-1500 area limits remain comparable
        # with our earlier v3 tests.
        sx = ANALYSIS_WIDTH / float(tray.shape[1])
        sy = ANALYSIS_HEIGHT / float(tray.shape[0])

        ax1 = int(nx1 * sx)
        ay1 = int(ny1 * sy)
        ax2 = int(nx2 * sx)
        ay2 = int(ny2 * sy)

        estimated_area, area_contour = estimate_pill_area(
            analysis_tray,
            analysis_mask,
            (ax1, ay1, ax2 - ax1, ay2 - ay1)
        )

        # If a foreground contour can be measured, enforce the
        # physical pill-area limit. This is deliberately based
        # on contour area rather than YOLO box area.
        if estimated_area > 0:
            if estimated_area < MIN_PILL_AREA:
                continue

            if estimated_area > MAX_PILL_AREA:
                continue

            hull = cv2.convexHull(area_contour)
            hull_area = cv2.contourArea(hull)

            if hull_area <= 0:
                continue

            solidity = estimated_area / hull_area
            if solidity < MIN_SOLIDITY:
                continue
        else:
            # If the color/foreground estimator cannot isolate a
            # contour, keep the YOLO candidate rather than making
            # the detector dependent on pill color.
            estimated_area = 0.0

        candidates.append({
            "confidence": confidence,
            "bbox": (nx1, ny1, nw, nh),
            "area": estimated_area
        })

    if not candidates:
        cv2.putText(
            frame,
            "NO VALID PILL",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        return frame, False, 0, 0.0

    # --------------------------------------------------------
    # 5. Draw EVERY candidate that passed all filters.
    #    This is what allows us to detect multiple pills.
    # --------------------------------------------------------
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    # Highest-confidence candidate is kept as the representative
    # area value for the stability wrapper/debug output.
    best = max(candidates, key=lambda c: c["confidence"])
    best_area = best["area"]

    for pill_number, candidate in enumerate(
        sorted(candidates, key=lambda c: c["confidence"], reverse=True),
        start=1
    ):
        nx, ny, nw, nh = candidate["bbox"]
        confidence = candidate["confidence"]
        estimated_area = candidate["area"]

        frame_x1 = int(crop_x1 + nx)
        frame_y1 = int(crop_y1 + ny)
        frame_x2 = int(crop_x1 + nx + nw)
        frame_y2 = int(crop_y1 + ny + nh)

        cv2.rectangle(
            frame,
            (frame_x1, frame_y1),
            (frame_x2, frame_y2),
            (0, 255, 0),
            2
        )

        label = f"PILL {pill_number}: {confidence:.2f}"
        if estimated_area > 0:
            label += f" A:{estimated_area:.0f}"
        else:
            label += " A:N/A"

        cv2.putText(
            frame,
            label,
            (frame_x1, max(25, frame_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    pill_count = len(candidates)

    cv2.putText(
        frame,
        "PILL DETECTED",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"PILLS DETECTED: {pill_count}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return frame, True, pill_count, best_area

# ============================================================
# STABILITY WRAPPER
# ============================================================

pill_detection_counter = 0
last_confirmed_count = 0
last_confirmed_area = 0.0


def process_frame(frame):
    global pill_detection_counter
    global last_confirmed_count
    global last_confirmed_area

    frame, detected, pill_count, area = detect_pill(frame)

    if detected and pill_count > 0:
        pill_detection_counter += 1
    else:
        pill_detection_counter = 0
        last_confirmed_count = 0
        last_confirmed_area = 0.0

    confirmed = pill_detection_counter >= REQUIRED_DETECTION_FRAMES

    if confirmed:
        last_confirmed_count = pill_count
        last_confirmed_area = area

        cv2.putText(
            frame,
            f"STABLE: {last_confirmed_count} PILL(S)",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            f"PILL FRAMES: {pill_detection_counter}/{REQUIRED_DETECTION_FRAMES}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    return frame, confirmed, last_confirmed_count, last_confirmed_area

# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():
    while True:
        frame = camera.capture_array()

        # Picamera2 gives RGB888, while OpenCV uses BGR.
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        frame, detected, pill_count, area = process_frame(frame)

        cv2.putText(
            frame,
            f"CAMERA: {frame.shape[1]}x{frame.shape[0]}",
            (20, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92]
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
# WEB PAGE
# ============================================================

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Pill Detection + Dynamic Tray Test</title>
        <style>
            body {
                font-family: Arial;
                background: #111;
                color: white;
                text-align: center;
            }
            img {
                width: 90%;
                max-width: 1000px;
                border: 2px solid white;
            }
        </style>
    </head>
    <body>
        <h1>YOLO Pill Detection + Dynamic Tray</h1>
        <img src="/video">
    </body>
    </html>
    """

# ============================================================
# VIDEO ROUTE
# ============================================================

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print()
    print("========================================")
    print(" YOLO + DYNAMIC TRAY MULTI-PILL TEST")
    print("========================================")
    print()
    print("Open in browser:")
    print("http://<PI-IP>:5001")
    print()
    print(f"MAX_PILL_AREA = {MAX_PILL_AREA}")
    print(f"REQUIRED_DETECTION_FRAMES = {REQUIRED_DETECTION_FRAMES}")
    print()

    app.run(
        host="0.0.0.0",
        port=5001,
        threaded=True
    )
