import cv2
import os
import time
import json
import platform
import subprocess
import threading
import queue
import sys
from pathlib import Path
from datetime import datetime
from collections import deque, Counter

import numpy as np
from ultralytics import YOLO


# ==========================================================
# PROJECT PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent if BASE_DIR.parent.name == "integration" else BASE_DIR.parent
INTEGRATION_DIR = PROJECT_ROOT / "integration"
if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))

try:
    from rashmi_tts_module.preference_tts import (
        preferences_from_env,
        speak_preference_voice,
    )
except Exception:
    speak_preference_voice = None
    preferences_from_env = None


def resolve_model_path() -> Path:
    env_path = os.getenv("YOLO_MODEL_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            BASE_DIR / "best.pt",
            PROJECT_ROOT / "manoj_camera_module" / "best.pt",
            BASE_DIR.parent / "manoj_camera_module" / "best.pt",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else BASE_DIR / "best.pt"


MODEL_PATH = resolve_model_path()
CONFIG_PATH = BASE_DIR / "guidance_config.json"
CAPTURE_DIR = BASE_DIR / "captured_pages"

CAPTURE_DIR.mkdir(exist_ok=True)


# ==========================================================
# CAMERA SETTINGS
# USB webcam on Raspberry Pi: CAMERA_INDEX=0 or 1
# Do not import picamera2 (that is Pi Camera Module 3 only).
# ==========================================================
_IS_LINUX = platform.system().lower() != "windows"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "15" if _IS_LINUX else "30"))
CAMERA_FLIP = os.getenv(
    "CAMERA_FLIP",
    "0" if _IS_LINUX else "1",
).strip().lower() in {"1", "true", "yes", "on"}
PREVIEW_MAX_WIDTH = int(os.getenv("PREVIEW_MAX_WIDTH", "640"))


# ==========================================================
# YOLO SETTINGS
# ==========================================================
YOLO_CONFIDENCE = 0.25
YOLO_FALLBACK_CONFIDENCE = 0.12
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "320" if _IS_LINUX else "640"))
YOLO_SKIP_FRAMES = int(os.getenv("YOLO_SKIP_FRAMES", "2" if _IS_LINUX else "0"))
PAGE_CLASS_NAME = "page"
MIN_PAGE_AREA_RATIO = 0.01
MAX_PAGE_AREA_RATIO = 0.99


# ==========================================================
# NEXT MODULE SETTINGS
# ==========================================================
# If your next module file name is different, change this.
# Example: "ocr_module.py", "image_description.py", "main.py"
NEXT_MODULE_SCRIPT = "next_module.py"

# If you do not want to run next module automatically, set this to False.
RUN_NEXT_MODULE_AUTOMATICALLY = True


# ==========================================================
# GUIDANCE CONFIG
# ==========================================================
DEFAULT_CONFIG = {
    # Typical full-page fill in the camera frame
    "target_area_ratio": 0.42,

    # 0.25 means 25% size difference allowed
    "distance_tolerance": 0.25,

    # Center tolerance
    "center_tolerance_x": 0.12,
    "center_tolerance_y": 0.12,

    # Full page must sit inside the frame with this margin
    "edge_margin": 0.04,

    # Do not capture a tiny distant page
    "min_full_page_area": 0.28,
    "min_width_fill": 0.28,
    "min_height_fill": 0.42,

    # Detection quality
    "minimum_confidence": 0.25
}


# ==========================================================
# AUTO CAPTURE SETTINGS
# ==========================================================
READY_STABLE_FRAMES = int(os.getenv("READY_STABLE_FRAMES", "8" if _IS_LINUX else "18"))
READY_HOLD_SECONDS = float(os.getenv("READY_HOLD_SECONDS", "2.0"))
NO_PAGE_ANNOUNCE_SECONDS = float(os.getenv("NO_PAGE_ANNOUNCE_SECONDS", "1.0"))
READY_MISS_TOLERANCE = int(os.getenv("READY_MISS_TOLERANCE", "5"))
CAPTURE_COOLDOWN_SECONDS = 8

CROP_MARGIN_RATIO = 0.10

STABILITY_WINDOW = 6
STABILITY_REQUIRED = 4


# ==========================================================
# VOICE SYSTEM
# ==========================================================
class VoiceSystem:
    """
    One speaker only. Each instruction finishes before the next starts.
    The camera loop never talks on the main thread, so guidance cannot overlap.
    """

    def __init__(self):
        self.voice_queue = queue.Queue()
        self._speaking = False
        self._lock = threading.Lock()
        self.last_text = ""
        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True
        )
        self.worker_thread.start()

    def _speak_text(self, text):
        if speak_preference_voice is not None:
            prefs = preferences_from_env() if preferences_from_env else None
            speak_preference_voice(text, prefs)
            return

        voice = os.getenv("RASHMI_VOICE", "female").strip().lower()
        pace = os.getenv("RASHMI_PACE", "normal").strip().lower()
        espeak_voice = "en+m3" if voice == "male" else "en+f3"
        pitch = "35" if voice == "male" else "72"
        speed = "140"
        if pace == "slow":
            speed = "115"
        elif pace == "fast":
            speed = "170"
        subprocess.run(
            ["espeak-ng", "-v", espeak_voice, "-s", speed, "-p", pitch, str(text)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def is_busy(self):
        with self._lock:
            speaking = self._speaking
        return speaking or not self.voice_queue.empty()

    def say(self, text, force=False):
        text = str(text or "").strip()
        if not text:
            return
        # force=True queues behind the current line. It never speaks in parallel.
        if not force and self.is_busy():
            return

        print("VOICE:", text)
        self.last_text = text
        self.voice_queue.put(text)

    def _worker(self):
        while True:
            text = self.voice_queue.get()

            if text is None:
                break

            with self._lock:
                self._speaking = True
            try:
                self._speak_text(text)
            except Exception as error:
                print("Voice error:", error)
            finally:
                with self._lock:
                    self._speaking = False


# ==========================================================
# CONFIG FUNCTIONS
# ==========================================================
def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as file:
                saved_config = json.load(file)

            config = DEFAULT_CONFIG.copy()
            config.update(saved_config)
            return config

        except Exception:
            return DEFAULT_CONFIG.copy()

    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=4)


def calibrate_correct_position(config, metrics):
    """
    Press C when the page is in the correct position.
    This saves your correct page size for this camera.
    """
    config["target_area_ratio"] = float(metrics["area_ratio"])
    save_config(config)

    print("\n====================================")
    print("CALIBRATION SAVED")
    print("Target Area Ratio:", round(config["target_area_ratio"], 3))
    print("====================================\n")


# ==========================================================
# CAMERA FUNCTION
# ==========================================================
def _configure_capture(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # USB webcams blur if autofocus is forced off.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, int(os.getenv("CAMERA_AUTOFOCUS", "1")))
    for _ in range(3):
        cap.read()
    print(
        "Camera configured:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "yolo_imgsz",
        YOLO_IMAGE_SIZE,
        "skip",
        YOLO_SKIP_FRAMES,
    )
    return cap


def open_camera(camera_index):
    """
    USB webcam via OpenCV.
    Windows: DirectShow. Raspberry Pi / Linux: V4L2 (/dev/video0, video1, ...).
    """
    camera_index = int(os.getenv("CAMERA_INDEX", camera_index))
    system_name = platform.system().lower()
    print("Opening USB camera index:", camera_index)

    cap = None

    if "windows" in system_name:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            for fallback_index in range(0, 5):
                if fallback_index == camera_index:
                    continue
                print("Trying USB camera index:", fallback_index)
                cap = cv2.VideoCapture(fallback_index, cv2.CAP_V4L2)
                if cap.isOpened():
                    print("Opened USB camera index:", fallback_index)
                    break

    if cap is None or not cap.isOpened():
        print("ERROR: USB camera not opened.")
        print("On Pi run: ls /dev/video*")
        print("Then retry with: export CAMERA_INDEX=1")
        return cap

    return _configure_capture(cap)


# ==========================================================
# YOLO PAGE DETECTION
# ==========================================================
def get_class_name(model_names, class_id):
    if isinstance(model_names, dict):
        return model_names.get(class_id, str(class_id))

    if isinstance(model_names, list):
        if class_id < len(model_names):
            return model_names[class_id]

    return str(class_id)


def _pick_best_page_box(result, frame_width, frame_height, model_names):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_detection = None
    best_score = 0

    for box in result.boxes:
        confidence = float(box.conf[0])

        class_id = int(box.cls[0])
        class_name = get_class_name(model_names, class_id)

        if PAGE_CLASS_NAME.lower() not in class_name.lower():
            if len(model_names) > 1:
                continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            continue

        area_ratio = (box_width * box_height) / (frame_width * frame_height)

        if area_ratio < MIN_PAGE_AREA_RATIO or area_ratio > MAX_PAGE_AREA_RATIO:
            continue

        if confidence > best_score:
            best_score = confidence
            best_detection = {
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name
            }

    return best_detection


def detect_page_with_yolo(model, frame, model_names):
    frame_height, frame_width = frame.shape[:2]
    confidences = (YOLO_CONFIDENCE,) if _IS_LINUX else (
        YOLO_CONFIDENCE,
        YOLO_FALLBACK_CONFIDENCE,
    )

    for confidence in confidences:
        results = model.predict(
            frame,
            imgsz=YOLO_IMAGE_SIZE,
            conf=confidence,
            verbose=False,
            max_det=1,
        )

        detection = _pick_best_page_box(
            results[0],
            frame_width,
            frame_height,
            model_names
        )

        if detection is not None:
            return detection

    return None


# ==========================================================
# METRIC CALCULATION
# ==========================================================
def calculate_metrics(detection, frame_width, frame_height):
    x1, y1, x2, y2 = detection["box"]

    box_width = x2 - x1
    box_height = y2 - y1

    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    offset_x = (box_center_x - frame_center_x) / frame_width
    offset_y = (box_center_y - frame_center_y) / frame_height

    area_ratio = (box_width * box_height) / (frame_width * frame_height)

    edge_left = x1 / frame_width
    edge_right = (frame_width - x2) / frame_width
    edge_top = y1 / frame_height
    edge_bottom = (frame_height - y2) / frame_height

    width_fill = box_width / frame_width
    height_fill = box_height / frame_height

    return {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "area_ratio": area_ratio,
        "edge_left": edge_left,
        "edge_right": edge_right,
        "edge_top": edge_top,
        "edge_bottom": edge_bottom,
        "width_fill": width_fill,
        "height_fill": height_fill
    }


# ==========================================================
# GUIDANCE DECISION
# ==========================================================
def decide_guidance(detection, metrics, config):
    """
    Speak how to get the whole page in view.
    Capture is allowed only when the full page is inside the frame.
    """
    confidence = detection["confidence"]

    if confidence < config["minimum_confidence"]:
        return "Show the full page clearly", False

    offset_x = metrics["offset_x"]
    offset_y = metrics["offset_y"]
    area_ratio = metrics["area_ratio"]

    target_area = config["target_area_ratio"]
    distance_tolerance = config["distance_tolerance"]
    center_tolerance_x = config["center_tolerance_x"]
    center_tolerance_y = config["center_tolerance_y"]
    edge_margin = config["edge_margin"]
    min_area = config.get("min_full_page_area", 0.28)
    min_width = config.get("min_width_fill", 0.28)
    min_height = config.get("min_height_fill", 0.42)

    too_small_area = max(min_area, target_area * (1 - distance_tolerance))
    too_large_area = min(0.85, target_area * (1 + distance_tolerance))

    clipped = []
    if metrics["edge_left"] < edge_margin:
        clipped.append("left")
    if metrics["edge_right"] < edge_margin:
        clipped.append("right")
    if metrics["edge_top"] < edge_margin:
        clipped.append("top")
    if metrics["edge_bottom"] < edge_margin:
        clipped.append("bottom")

    # Partial page: never capture until every edge is inside the frame
    if clipped:
        if len(clipped) >= 2 or area_ratio > target_area:
            return "Move farther so the full page fits", False
        if clipped == ["left"]:
            return "Move right. The left side is cut off", False
        if clipped == ["right"]:
            return "Move left. The right side is cut off", False
        if clipped == ["top"]:
            return "Move down. The top of the page is cut off", False
        if clipped == ["bottom"]:
            return "Move up. The bottom of the page is cut off", False
        return "Move farther so the full page fits", False

    if offset_x < -center_tolerance_x:
        return "Move left to center the page", False

    if offset_x > center_tolerance_x:
        return "Move right to center the page", False

    if offset_y < -center_tolerance_y:
        return "Move up to center the page", False

    if offset_y > center_tolerance_y:
        return "Move down to center the page", False

    if (
        area_ratio < too_small_area
        or metrics["width_fill"] < min_width
        or metrics["height_fill"] < min_height
    ):
        return "Move closer until the full page is visible", False

    if area_ratio > too_large_area:
        return "Move farther so the full page fits", False

    return "Full page visible. Hold still", True


# ==========================================================
# STABLE VOICE FILTER
# ==========================================================
def get_stable_voice_text(history, instruction):
    history.append(instruction)

    if len(history) < STABILITY_WINDOW:
        return None

    most_common, count = Counter(history).most_common(1)[0]

    if count >= STABILITY_REQUIRED:
        return most_common

    return None


# ==========================================================
# CAPTURE FUNCTIONS
# ==========================================================
def crop_page_with_margin(frame, detection):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = detection["box"]

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = int(box_width * CROP_MARGIN_RATIO)
    margin_y = int(box_height * CROP_MARGIN_RATIO)

    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(frame_width, x2 + margin_x)
    crop_y2 = min(frame_height, y2 + margin_y)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    return crop


def save_capture(frame, detection):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_image_path = CAPTURE_DIR / f"page_full_{timestamp}.jpg"
    crop_image_path = CAPTURE_DIR / f"page_crop_{timestamp}.jpg"

    crop = crop_page_with_margin(frame, detection)

    cv2.imwrite(str(full_image_path), frame)
    cv2.imwrite(str(crop_image_path), crop)

    # Save latest path for next module
    latest_path_file = BASE_DIR / "latest_captured_image.txt"
    with open(latest_path_file, "w") as file:
        file.write(str(crop_image_path))

    print("\n====================================")
    print("IMAGE CAPTURED")
    print("Full image:", full_image_path)
    print("Crop image:", crop_image_path)
    print("Latest path saved in:", latest_path_file)
    print("====================================\n")

    return crop_image_path


def send_to_next_module(image_path):
    next_module_path = BASE_DIR / NEXT_MODULE_SCRIPT

    if not RUN_NEXT_MODULE_AUTOMATICALLY:
        print("Next module auto run is disabled.")
        return

    if not next_module_path.exists():
        print("Next module not found:", next_module_path)
        print("Image is saved. You can send this path manually:", image_path)
        return

    print("Sending image to next module:", next_module_path)

    try:
        subprocess.run(
            [
                sys.executable,
                str(next_module_path),
                str(image_path)
            ],
            cwd=str(BASE_DIR),
            check=False,
        )
    except Exception as error:
        print("Error sending image to next module:", error)


# ==========================================================
# DRAW UI
# ==========================================================
def draw_ui(frame, detection, metrics, instruction, ready_count, config, ready_hold_seconds=0.0):
    frame_height, frame_width = frame.shape[:2]

    # Top black background
    cv2.rectangle(frame, (0, 0), (frame_width, 100), (0, 0, 0), -1)

    color = (0, 255, 0) if "full page visible" in instruction.lower() else (0, 0, 255)

    cv2.putText(
        frame,
        instruction,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        color,
        3
    )

    # Center lines
    cv2.line(
        frame,
        (frame_width // 2, 0),
        (frame_width // 2, frame_height),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (0, frame_height // 2),
        (frame_width, frame_height // 2),
        (255, 255, 255),
        1
    )

    # Center tolerance box
    tolerance_x = int(config["center_tolerance_x"] * frame_width)
    tolerance_y = int(config["center_tolerance_y"] * frame_height)

    cv2.rectangle(
        frame,
        (frame_width // 2 - tolerance_x, frame_height // 2 - tolerance_y),
        (frame_width // 2 + tolerance_x, frame_height // 2 + tolerance_y),
        (255, 255, 0),
        2
    )

    if detection is not None:
        x1, y1, x2, y2 = detection["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'

        cv2.putText(
            frame,
            label,
            (x1, max(125, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

    y = 130

    if metrics is not None:
        debug_lines = [
            f'Area Ratio: {metrics["area_ratio"]:.3f}',
            f'Target Area: {config["target_area_ratio"]:.3f}',
            f'Offset X: {metrics["offset_x"]:.3f}',
            f'Offset Y: {metrics["offset_y"]:.3f}',
            f'Width Fill: {metrics["width_fill"]:.3f}',
            f'Height Fill: {metrics["height_fill"]:.3f}',
            f'Hold: {ready_hold_seconds:.1f}/{READY_HOLD_SECONDS:.1f}s',
        ]
    else:
        debug_lines = [
            "No page detected",
            "Fill ~30-60% of frame with a flat page",
            "Use good lighting, hold steady"
        ]

    for line in debug_lines:
        cv2.putText(
            frame,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )
        y += 30

    # Bottom help bar
    cv2.rectangle(
        frame,
        (0, frame_height - 50),
        (frame_width, frame_height),
        (0, 0, 0),
        -1
    )

    help_text = "Q = Quit | C = Calibrate correct position | V = Test voice"

    cv2.putText(
        frame,
        help_text,
        (30, frame_height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return frame


# ==========================================================
# MAIN FUNCTION
# ==========================================================
def main():
    print("==========================================")
    print("YOLO PAGE GUIDANCE + AUTO CAPTURE STARTED")
    print("==========================================")

    if not MODEL_PATH.exists():
        print("ERROR: best.pt not found.")
        print("Looked for:", MODEL_PATH)
        print("Expected on the Pi: /home/readingassistant/R26-IT-041/manoj_camera_module/best.pt")
        print("Or set YOLO_MODEL_PATH to the full path of best.pt.")
        return

    config = load_config()

    print("Loading YOLO model:", MODEL_PATH)
    print("This can take 10 to 30 seconds on Raspberry Pi. Please wait.")
    t_load = time.perf_counter()
    model = YOLO(str(MODEL_PATH), task="detect")
    model_names = model.names
    print(f"YOLO model loaded in {time.perf_counter() - t_load:.1f}s")

    print("Model classes:", model_names)

    print("Opening camera index:", CAMERA_INDEX)
    cap = open_camera(CAMERA_INDEX)

    if cap is None or not cap.isOpened():
        print("ERROR: USB camera not opened.")
        print("Unplug Pi Camera Module 3 ribbon if you switched to USB.")
        print("Run: ls /dev/video*")
        print("Then: export CAMERA_INDEX=0   or   export CAMERA_INDEX=1")
        return

    voice = VoiceSystem()
    voice.say(
        "Point the camera at the page. I will guide you until the full page is visible."
    )

    voice_history = deque(maxlen=STABILITY_WINDOW)

    ready_count = 0
    ready_since = None
    ready_miss_count = 0
    no_page_since = None
    last_capture_time = 0
    session_has_captured = False
    last_logged_instruction = ""
    last_log_time = 0
    last_detection = None
    hold_announced = False
    frame_index = 0
    single_capture = os.getenv("SINGLE_CAPTURE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    while True:
        cap.grab()
        ret, frame = cap.read()

        if not ret:
            print("Cannot read camera frame.")
            break

        if CAMERA_FLIP:
            frame = cv2.flip(frame, 1)

        clean_frame = frame.copy()

        frame_height, frame_width = frame.shape[:2]
        frame_index += 1
        run_yolo = YOLO_SKIP_FRAMES <= 0 or (
            frame_index % (YOLO_SKIP_FRAMES + 1) == 0
        )

        if run_yolo:
            last_detection = detect_page_with_yolo(
                model,
                frame,
                model_names
            )

        detection = last_detection
        current_time = time.time()

        if detection is None:
            metrics = None
            instruction = "No page detected"
            is_ready = False
            hold_announced = False
            ready_miss_count += 1
            if ready_miss_count >= READY_MISS_TOLERANCE:
                ready_count = 0
                ready_since = None
            if no_page_since is None:
                no_page_since = current_time

        else:
            no_page_since = None
            metrics = calculate_metrics(
                detection,
                frame_width,
                frame_height
            )

            instruction, is_ready = decide_guidance(
                detection,
                metrics,
                config
            )

            if is_ready:
                ready_miss_count = 0
                ready_count += 1
                if ready_since is None:
                    ready_since = current_time
            else:
                hold_announced = False
                ready_miss_count += 1
                if ready_miss_count >= READY_MISS_TOLERANCE:
                    ready_count = 0
                    ready_since = None

            hold_seconds = (current_time - ready_since) if ready_since else 0.0
            if instruction != last_logged_instruction or current_time - last_log_time > 2.0:
                print(
                    f'Hold: {hold_seconds:.1f}/{READY_HOLD_SECONDS:.1f}s | '
                    f'Guide: {instruction}'
                )
                last_logged_instruction = instruction
                last_log_time = current_time

        # Repeat guidance until the page is fully visible.
        # Speak only when the previous instruction has finished.
        if not is_ready:
            should_speak = True
            if instruction == "No page detected":
                should_speak = (
                    no_page_since is not None
                    and current_time - no_page_since >= NO_PAGE_ANNOUNCE_SECONDS
                )
            else:
                stable_text = get_stable_voice_text(voice_history, instruction)
                should_speak = stable_text is not None
                if should_speak:
                    instruction = stable_text

            if should_speak and not voice.is_busy():
                voice.say(instruction)
        elif not hold_announced and not voice.is_busy():
            voice.say("Full page visible. Hold still.")
            hold_announced = True

        hold_seconds = (current_time - ready_since) if ready_since else 0.0
        cooldown_ok = (
            not session_has_captured
            or current_time - last_capture_time > CAPTURE_COOLDOWN_SECONDS
        )

        if (
            detection is not None and
            is_ready and
            ready_since is not None and
            hold_seconds >= READY_HOLD_SECONDS and
            cooldown_ok and
            not voice.is_busy()
        ):
            voice.say("Capturing now.", force=True)

            captured_image_path = save_capture(
                clean_frame,
                detection
            )

            send_to_next_module(captured_image_path)

            voice.say("Page captured successfully.", force=True)

            last_capture_time = current_time
            session_has_captured = True
            ready_count = 0
            ready_since = None
            hold_announced = False

            if single_capture:
                print("Single capture done. Closing camera.")
                break

        display_frame = draw_ui(
            frame,
            detection,
            metrics,
            instruction,
            ready_count,
            config,
            ready_hold_seconds=hold_seconds,
        )

        if display_frame.shape[1] > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / float(display_frame.shape[1])
            display_frame = cv2.resize(
                display_frame,
                (
                    PREVIEW_MAX_WIDTH,
                    max(1, int(display_frame.shape[0] * scale)),
                ),
            )

        cv2.imshow(
            "YOLO Page Guidance + Auto Capture",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("v"):
            voice.say("Voice test working")

        if key == ord("c"):
            if detection is not None and metrics is not None:
                calibrate_correct_position(config, metrics)
                voice.say("Calibration saved")
            else:
                print("Cannot calibrate. No page detected.")
                voice.say("No page detected")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()