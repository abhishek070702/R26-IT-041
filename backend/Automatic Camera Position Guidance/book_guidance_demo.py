import cv2
import numpy as np
import time
import math
import os
import subprocess
from collections import deque, Counter


# ==========================================================
# BASIC SETTINGS
# ==========================================================

CAMERA_INDEX = 0

FRAME_W = 640
FRAME_H = 480

CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

# Movement tolerance
X_TOL = 80
Y_TOL = 65

# Tilt tolerance
ANGLE_TOL = 12

# Page size limits
MIN_AREA_RATIO = 0.08
MAX_AREA_RATIO = 0.82

# Voice timing
VOICE_MIN_GAP = 1.8
VOICE_REPEAT_GAP = 4.0

# Auto capture
CENTERED_FRAMES_REQUIRED = 25

CAPTURE_FILE = "captured_page.jpg"


# ==========================================================
# WINDOWS VOICE FEEDBACK
# ==========================================================

class VoiceFeedback:
    def __init__(self):
        self.last_spoken = ""
        self.last_spoken_time = 0
        self.current_process = None

    def request(self, text, force=False):
        now = time.time()

        same_instruction = text == self.last_spoken
        time_since_last = now - self.last_spoken_time

        if not force and same_instruction and time_since_last < VOICE_REPEAT_GAP:
            return

        if not force and not same_instruction and time_since_last < VOICE_MIN_GAP:
            return

        # Do not start new speech while previous speech is running
        if self.current_process is not None:
            if self.current_process.poll() is None:
                return

        self.speak(text)

        self.last_spoken = text
        self.last_spoken_time = time.time()

    def speak(self, text):
        print("Voice:", text)

        if os.name != "nt":
            print("Voice is configured for Windows only in this demo.")
            return

        safe_text = text.replace("'", "''")

        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Rate = 0; "
            f"$speaker.Speak('{safe_text}');"
        )

        self.current_process = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

    def stop(self):
        if self.current_process is not None:
            try:
                self.current_process.terminate()
            except Exception:
                pass


voice = VoiceFeedback()


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def order_points(pts):
    """
    Order points:
    top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def calculate_angle(corners):
    """
    Calculate page tilt angle using the top edge.
    Negative angle = tilted left
    Positive angle = tilted right
    """
    tl, tr, br, bl = corners

    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]

    angle = math.degrees(math.atan2(dy, dx))

    if angle > 45:
        angle -= 90

    if angle < -45:
        angle += 90

    return angle


def calculate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ==========================================================
# IMAGE PROCESSING
# ==========================================================

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # Reduce noise while keeping edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    return denoised


def create_edge_binary(processed):
    edges = cv2.Canny(processed, 35, 120)

    kernel = np.ones((5, 5), np.uint8)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def create_bright_binary(processed):
    blur = cv2.GaussianBlur(processed, (5, 5), 0)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return binary


def create_adaptive_binary(processed):
    binary = cv2.adaptiveThreshold(
        processed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        8
    )

    kernel = np.ones((7, 7), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return binary


# ==========================================================
# PAGE DETECTION
# ==========================================================

def find_candidates(binary_image, method_name):
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    frame_area = FRAME_W * FRAME_H
    candidates = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area < frame_area * 0.015:
            continue

        hull = cv2.convexHull(contour)

        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        w, h = rect[1]

        if w <= 0 or h <= 0:
            continue

        rect_area = w * h
        area_ratio = rect_area / frame_area

        if area_ratio < 0.04:
            continue

        if area_ratio > 0.95:
            continue

        aspect_ratio = max(w, h) / min(w, h)

        # Page/open book should be rectangular
        if aspect_ratio < 1.05:
            continue

        if aspect_ratio > 5.0:
            continue

        fill_ratio = contour_area / rect_area

        if method_name == "edge":
            if fill_ratio < 0.05:
                continue
        else:
            if fill_ratio < 0.20:
                continue

        x_values = box[:, 0]
        y_values = box[:, 1]

        box_width = max(x_values) - min(x_values)
        box_height = max(y_values) - min(y_values)

        if box_width < 100 or box_height < 100:
            continue

        score = rect_area

        if method_name == "bright":
            score *= 1.15

        if method_name == "adaptive":
            score *= 0.95

        score *= min(fill_ratio + 0.5, 1.5)

        candidates.append({
            "score": score,
            "corners": order_points(box),
            "area": rect_area,
            "method": method_name
        })

    return candidates


def detect_page(frame):
    processed = preprocess_frame(frame)

    edge_binary = create_edge_binary(processed)
    bright_binary = create_bright_binary(processed)
    adaptive_binary = create_adaptive_binary(processed)

    candidates = []

    candidates.extend(find_candidates(edge_binary, "edge"))
    candidates.extend(find_candidates(bright_binary, "bright"))
    candidates.extend(find_candidates(adaptive_binary, "adaptive"))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c["score"], reverse=True)

    best = candidates[0]
    corners = best["corners"]

    center_x = int(np.mean(corners[:, 0]))
    center_y = int(np.mean(corners[:, 1]))

    angle = calculate_angle(corners)

    return {
        "corners": corners,
        "center": (center_x, center_y),
        "area": best["area"],
        "angle": angle,
        "method": best["method"]
    }


# ==========================================================
# GUIDANCE LOGIC
# ==========================================================

def get_instruction(page_info, brightness):
    if brightness < 35 and page_info is None:
        return "too dark"

    if page_info is None:
        return "no page detected"

    center_x, center_y = page_info["center"]
    area = page_info["area"]
    angle = page_info["angle"]

    frame_area = FRAME_W * FRAME_H
    area_ratio = area / frame_area

    dx = center_x - CENTER_X
    dy = center_y - CENTER_Y

    # Priority order
    if area_ratio < MIN_AREA_RATIO:
        return "move closer"

    if area_ratio > MAX_AREA_RATIO:
        return "move farther"

    if dx < -X_TOL:
        return "move right"

    if dx > X_TOL:
        return "move left"

    if dy < -Y_TOL:
        return "move down"

    if dy > Y_TOL:
        return "move up"

    if angle < -ANGLE_TOL:
        return "tilted left"

    if angle > ANGLE_TOL:
        return "tilted right"

    return "centered ready"


def smooth_instruction(history, new_instruction):
    history.append(new_instruction)

    counter = Counter(history)
    most_common_instruction, count = counter.most_common(1)[0]

    if count >= 3:
        return most_common_instruction

    return new_instruction


def perspective_crop(frame, corners):
    tl, tr, br, bl = corners

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width <= 0 or max_height <= 0:
        return None

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))

    return warped


# ==========================================================
# DRAWING FUNCTIONS
# ==========================================================

def draw_guides(frame):
    # Center cross
    cv2.line(
        frame,
        (CENTER_X - 45, CENTER_Y),
        (CENTER_X + 45, CENTER_Y),
        (255, 0, 0),
        2
    )

    cv2.line(
        frame,
        (CENTER_X, CENTER_Y - 45),
        (CENTER_X, CENTER_Y + 45),
        (255, 0, 0),
        2
    )

    # Acceptable center box
    cv2.rectangle(
        frame,
        (CENTER_X - X_TOL, CENTER_Y - Y_TOL),
        (CENTER_X + X_TOL, CENTER_Y + Y_TOL),
        (255, 0, 0),
        2
    )


def draw_page_info(frame, page_info):
    if page_info is None:
        return

    corners = page_info["corners"].astype(int)

    cv2.drawContours(
        frame,
        [corners],
        -1,
        (0, 255, 0),
        3
    )

    cx, cy = page_info["center"]

    cv2.circle(
        frame,
        (cx, cy),
        6,
        (0, 0, 255),
        -1
    )

    area_ratio = page_info["area"] / (FRAME_W * FRAME_H)

    cv2.putText(
        frame,
        f"Area: {area_ratio:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Angle: {page_info['angle']:.1f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Method: {page_info['method']}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():
    print("===================================")
    print("Book Guidance Demo Started")
    print("===================================")
    print("Controls:")
    print("q = quit")
    print("v = test voice")
    print("c = manual capture")
    print("-----------------------------------")
    print("Use white paper or open book page.")
    print("Use bright light.")
    print("Use dark/plain background.")
    print("===================================")

    if os.name == "nt":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Camera not found.")
        print("Try changing CAMERA_INDEX = 0 to 1 or 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # These may not work on every webcam, but they are safe
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    cap.set(cv2.CAP_PROP_CONTRAST, 50)
    cap.set(cv2.CAP_PROP_SHARPNESS, 50)

    instruction_history = deque(maxlen=6)
    centered_count = 0
    last_page_info = None

    voice.request("system started", force=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        brightness = calculate_brightness(frame)
        sharpness = calculate_sharpness(frame)

        page_info = detect_page(frame)
        last_page_info = page_info

        raw_instruction = get_instruction(page_info, brightness)
        instruction = smooth_instruction(instruction_history, raw_instruction)

        voice.request(instruction)

        draw_guides(frame)
        draw_page_info(frame, page_info)

        cv2.putText(
            frame,
            instruction,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Brightness: {brightness:.0f}",
            (20, FRAME_H - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Sharpness: {sharpness:.0f}",
            (20, FRAME_H - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Auto capture when centered
        if instruction == "centered ready" and page_info is not None:
            centered_count += 1
        else:
            centered_count = 0

        if centered_count >= CENTERED_FRAMES_REQUIRED:
            cropped = perspective_crop(frame, page_info["corners"])

            if cropped is not None:
                cv2.imwrite(CAPTURE_FILE, cropped)
                print("Captured:", CAPTURE_FILE)
                voice.request("image captured", force=True)

            centered_count = 0

        cv2.imshow("Book Guidance Demo", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("v"):
            voice.request("voice test working", force=True)

        if key == ord("c"):
            if last_page_info is not None:
                cropped = perspective_crop(frame, last_page_info["corners"])

                if cropped is not None:
                    cv2.imwrite(CAPTURE_FILE, cropped)
                    print("Manual capture saved:", CAPTURE_FILE)
                    voice.request("image captured", force=True)
            else:
                print("No page detected for manual capture.")
                voice.request("no page detected", force=True)

    cap.release()
    cv2.destroyAllWindows()
    voice.stop()


if __name__ == "__main__":
    main()