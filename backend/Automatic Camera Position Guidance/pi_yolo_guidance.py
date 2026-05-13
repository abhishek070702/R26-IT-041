import cv2
import time
import json
import subprocess
import threading
import queue
import sys
from pathlib import Path
from datetime import datetime
from collections import deque, Counter

import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2


# ==========================================================
# PROJECT PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
CONFIG_PATH = BASE_DIR / "guidance_config.json"
CAPTURE_DIR = BASE_DIR / "captured_pages"

CAPTURE_DIR.mkdir(exist_ok=True)


# ==========================================================
# CAMERA SETTINGS FOR RASPBERRY PI
# ==========================================================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ==========================================================
# YOLO SETTINGS
# ==========================================================
YOLO_CONFIDENCE = 0.30
YOLO_IMAGE_SIZE = 416
PAGE_CLASS_NAME = "page"


# ==========================================================
# NEXT MODULE SETTINGS
# ==========================================================
NEXT_MODULE_SCRIPT = "next_module.py"
RUN_NEXT_MODULE_AUTOMATICALLY = True


# ==========================================================
# GUIDANCE CONFIG
# ==========================================================
DEFAULT_CONFIG = {
    "target_area_ratio": 0.45,
    "distance_tolerance": 0.20,
    "center_tolerance_x": 0.08,
    "center_tolerance_y": 0.08,
    "edge_margin": 0.03,
    "minimum_confidence": 0.30
}


# ==========================================================
# AUTO CAPTURE SETTINGS
# ==========================================================
READY_STABLE_FRAMES = 15
CAPTURE_COOLDOWN_SECONDS = 8

CROP_MARGIN_RATIO = 0.10

VOICE_COOLDOWN_SECONDS = 2.5
STABILITY_WINDOW = 8
STABILITY_REQUIRED = 5

SHOW_WINDOW = True


# ==========================================================
# VOICE SYSTEM FOR RASPBERRY PI
# ==========================================================
class VoiceSystem:
    def __init__(self):
        self.voice_queue = queue.Queue(maxsize=1)
        self.last_text = ""
        self.last_time = 0

        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True
        )
        self.worker_thread.start()

    def _speak_pi(self, text):
        subprocess.run(
            ["espeak", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def _worker(self):
        while True:
            text = self.voice_queue.get()

            if text is None:
                break

            try:
                self._speak_pi(text)
            except Exception as error:
                print("Voice error:", error)

    def say(self, text):
        current_time = time.time()

        if text == self.last_text and current_time - self.last_time < VOICE_COOLDOWN_SECONDS:
            return

        if current_time - self.last_time < VOICE_COOLDOWN_SECONDS:
            return

        while not self.voice_queue.empty():
            try:
                self.voice_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.voice_queue.put_nowait(text)
            self.last_text = text
            self.last_time = current_time
            print("VOICE:", text)
        except queue.Full:
            pass


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
    config["target_area_ratio"] = float(metrics["area_ratio"])
    save_config(config)

    print("\n====================================")
    print("CALIBRATION SAVED")
    print("Target Area Ratio:", round(config["target_area_ratio"], 3))
    print("====================================\n")


# ==========================================================
# CAMERA FUNCTIONS
# ==========================================================
def open_pi_camera():
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(2)

    return picam2


def get_pi_camera_frame(picam2):
    rgb_frame = picam2.capture_array()

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    return frame


# ==========================================================
# CAMERA IMAGE IMPROVEMENT
# ==========================================================
def improve_camera_frame(frame):
    """
    Reduce noise and improve contrast before YOLO detection.
    Useful for Camera Module 2 or low-light camera input.
    """

    frame = cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        5,
        5,
        7,
        21
    )

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame


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


def detect_page_with_yolo(model, frame, model_names):
    frame_height, frame_width = frame.shape[:2]

    improved_frame = improve_camera_frame(frame)

    results = model(
        improved_frame,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    result = results[0]

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

        if area_ratio < 0.03 or area_ratio > 0.97:
            continue

        score = confidence * area_ratio

        if score > best_score:
            best_score = score
            best_detection = {
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name
            }

    return best_detection


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
    confidence = detection["confidence"]

    if confidence < config["minimum_confidence"]:
        return "Show the page clearly", False

    offset_x = metrics["offset_x"]
    offset_y = metrics["offset_y"]
    area_ratio = metrics["area_ratio"]

    target_area = config["target_area_ratio"]
    distance_tolerance = config["distance_tolerance"]

    center_tolerance_x = config["center_tolerance_x"]
    center_tolerance_y = config["center_tolerance_y"]

    edge_margin = config["edge_margin"]

    too_small_area = target_area * (1 - distance_tolerance)
    too_large_area = target_area * (1 + distance_tolerance)

    page_touches_edge = (
        metrics["edge_left"] < edge_margin or
        metrics["edge_right"] < edge_margin or
        metrics["edge_top"] < edge_margin or
        metrics["edge_bottom"] < edge_margin
    )

    if page_touches_edge and area_ratio > target_area * 0.85:
        return "Move farther", False

    if offset_x < -center_tolerance_x:
        return "Move camera left", False

    if offset_x > center_tolerance_x:
        return "Move camera right", False

    if offset_y < -center_tolerance_y:
        return "Move camera up", False

    if offset_y > center_tolerance_y:
        return "Move camera down", False

    if area_ratio < too_small_area:
        return "Move closer", False

    if area_ratio > too_large_area:
        return "Move farther", False

    return "Ready. Hold steady", True


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
        subprocess.Popen(
            [
                sys.executable,
                str(next_module_path),
                str(image_path)
            ],
            cwd=str(BASE_DIR)
        )
    except Exception as error:
        print("Error sending image to next module:", error)


# ==========================================================
# DRAW UI
# ==========================================================
def draw_ui(frame, detection, metrics, instruction, ready_count, config):
    frame_height, frame_width = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (frame_width, 100), (0, 0, 0), -1)

    color = (0, 255, 0) if "Ready" in instruction else (0, 0, 255)

    cv2.putText(
        frame,
        instruction,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        color,
        3
    )

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
            f'Ready Frames: {ready_count}/{READY_STABLE_FRAMES}',
        ]
    else:
        debug_lines = [
            "No page detected",
            "Show the page clearly"
        ]

    for line in debug_lines:
        cv2.putText(
            frame,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        y += 26

    cv2.rectangle(
        frame,
        (0, frame_height - 50),
        (frame_width, frame_height),
        (0, 0, 0),
        -1
    )

    help_text = "Q = Quit | C = Calibrate | V = Voice Test"

    cv2.putText(
        frame,
        help_text,
        (30, frame_height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return frame


# ==========================================================
# MAIN FUNCTION
# ==========================================================
def main():
    print("==========================================")
    print("RASPBERRY PI YOLO PAGE GUIDANCE STARTED")
    print("==========================================")

    if not MODEL_PATH.exists():
        print("ERROR: best.pt not found.")
        print("Put best.pt in the same folder as this Python file.")
        return

    config = load_config()

    print("Loading YOLO model:", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))
    model_names = model.names

    print("Model classes:", model_names)

    print("Opening Raspberry Pi camera...")
    picam2 = open_pi_camera()

    voice = VoiceSystem()
    voice.say("Camera guidance started")

    voice_history = deque(maxlen=STABILITY_WINDOW)

    ready_count = 0
    last_capture_time = 0

    while True:
        frame = get_pi_camera_frame(picam2)

        clean_frame = frame.copy()

        frame_height, frame_width = frame.shape[:2]

        detection = detect_page_with_yolo(
            model,
            frame,
            model_names
        )

        if detection is None:
            metrics = None
            instruction = "No page detected"
            is_ready = False
            ready_count = 0

        else:
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
                ready_count += 1
            else:
                ready_count = 0

            print(
                f'Conf: {detection["confidence"]:.2f} | '
                f'Area: {metrics["area_ratio"]:.3f} | '
                f'Target: {config["target_area_ratio"]:.3f} | '
                f'X: {metrics["offset_x"]:.3f} | '
                f'Y: {metrics["offset_y"]:.3f} | '
                f'Ready: {ready_count}/{READY_STABLE_FRAMES} | '
                f'Guide: {instruction}'
            )

        stable_text = get_stable_voice_text(
            voice_history,
            instruction
        )

        if stable_text is not None:
            voice.say(stable_text)

        current_time = time.time()

        if (
            detection is not None and
            is_ready and
            ready_count >= READY_STABLE_FRAMES and
            current_time - last_capture_time > CAPTURE_COOLDOWN_SECONDS
        ):
            voice.say("Capturing page")

            captured_image_path = save_capture(
                clean_frame,
                detection
            )

            send_to_next_module(captured_image_path)

            voice.say("Page captured successfully")

            last_capture_time = current_time
            ready_count = 0

        if SHOW_WINDOW:
            display_frame = draw_ui(
                frame,
                detection,
                metrics,
                instruction,
                ready_count,
                config
            )

            cv2.imshow(
                "Raspberry Pi YOLO Page Guidance",
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

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()