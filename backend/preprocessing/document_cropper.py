"""
Local document / book / page cropper.

Detects the main reading-material region in a camera capture and crops it
so OCR, document ID, and BLIP focus on the page — not hands, bottles, walls.

No OpenAI / Google / external APIs. OpenCV only.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CROP_DIR = os.path.join(PROJECT_DIR, "backend", "outputs", "crops")
os.makedirs(CROP_DIR, exist_ok=True)

# Minimum share of the frame the page should cover to count as a valid crop.
MIN_AREA_RATIO = 0.08
MAX_DETECT_DIM = 1200
# Downstream OCR/BLIP processing width (keeps aspect ratio).
MAX_PROCESS_WIDTH = 1200


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_corners(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 40 or max_height < 40:
        raise ValueError("Perspective crop too small")

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _resize_for_detect(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_DETECT_DIM:
        return image, 1.0
    scale = MAX_DETECT_DIM / float(longest)
    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _find_page_quad(
    gray: np.ndarray,
    image_area: float,
) -> Optional[np.ndarray]:
    """
    Find the best 4-point page contour in a grayscale detect image.
    Returns quad points in detect-image coordinates, or None.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Combine multiple edge passes for magazine/book covers with weak borders.
    edges_a = cv2.Canny(blurred, 50, 150)
    edges_b = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_or(edges_a, edges_b)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    adaptive = cv2.bitwise_not(adaptive)
    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2,
    )

    combined = cv2.bitwise_or(edges, adaptive)

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    best_quad = None
    best_area = 0.0
    height, width = gray.shape[:2]

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * MIN_AREA_RATIO:
            continue
        # Skip near-full-frame contours (image border, not the page).
        if area > image_area * 0.92:
            continue

        peri = cv2.arcLength(contour, True)
        approx = None
        for epsilon_factor in (0.02, 0.03, 0.015, 0.04):
            candidate = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
            if len(candidate) == 4 and cv2.isContourConvex(candidate):
                approx = candidate
                break
        if approx is None:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w >= width * 0.98 and h >= height * 0.98:
            continue

        if area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2).astype(np.float32)

    return best_quad


def _find_largest_bbox(
    gray: np.ndarray,
    image_area: float,
) -> Optional[tuple[int, int, int, int]]:
    """Fallback: largest rectangular contour bounding box."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    height, width = gray.shape[:2]

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * MIN_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        # Reject near-full-frame boxes (likely no real page found).
        if w >= width * 0.98 and h >= height * 0.98:
            continue
        if w < 40 or h < 40:
            continue
        return x, y, w, h

    return None


def _add_padding(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.02,
) -> tuple[int, int, int, int]:
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return x1, y1, x2 - x1, y2 - y1


def _save_crop(image: np.ndarray, source_path: str, suffix: str = "crop") -> str:
    base = os.path.splitext(os.path.basename(source_path))[0] or "page"
    out_name = f"{base}_{suffix}_{uuid.uuid4().hex[:10]}.jpg"
    out_path = os.path.join(CROP_DIR, out_name)
    ok = cv2.imwrite(out_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Failed to write crop image: {out_path}")
    return out_path


def resize_for_processing(
    image_bgr: np.ndarray,
    max_width: int = MAX_PROCESS_WIDTH,
) -> np.ndarray:
    """Resize image to max_width while keeping aspect ratio."""
    height, width = image_bgr.shape[:2]
    if width <= max_width:
        return image_bgr
    scale = max_width / float(width)
    new_w = max_width
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def optimize_processing_image(image_path: str) -> str:
    """
    Create an optimized copy for OCR / doc-ID / BLIP.
    Max width 1200px, aspect ratio preserved.
    Saves under backend/outputs/crops/.
    Returns optimized path, or original path on failure.
    """
    if not image_path or not os.path.isfile(image_path):
        return image_path

    try:
        image = cv2.imread(image_path)
        if image is None:
            return image_path

        h, w = image.shape[:2]
        if w <= MAX_PROCESS_WIDTH:
            print(
                f"[document_cropper] processing image already sized {w}x{h}; "
                f"reusing path={image_path}"
            )
            return image_path

        resized = resize_for_processing(image, MAX_PROCESS_WIDTH)
        rh, rw = resized.shape[:2]
        out_path = _save_crop(resized, image_path, suffix="proc")
        print(
            f"[document_cropper] optimized processing image: {w}x{h} -> {rw}x{rh}"
        )
        print(f"[document_cropper] processing image path: {out_path}")
        logger.info(
            "optimized processing image %sx%s -> %sx%s path=%s",
            w,
            h,
            rw,
            rh,
            out_path,
        )
        return out_path
    except Exception:
        logger.exception("optimize_processing_image failed for %r", image_path)
        return image_path


def crop_reading_material(image_path: str) -> str:
    """
    Crop the main document/book/page region from a camera image.

    Returns:
        Path to the cropped image on success, or the original image_path
        if detection/cropping fails (never raises to the caller).
    """
    print(f"[document_cropper] original image path: {image_path}")
    logger.info("crop_reading_material original path=%r", image_path)

    if not image_path or not os.path.isfile(image_path):
        print("[document_cropper] crop fail: image path missing")
        logger.warning("crop failed: missing path=%r", image_path)
        return image_path

    try:
        image = cv2.imread(image_path)
        if image is None:
            print("[document_cropper] crop fail: could not read image")
            logger.warning("crop failed: unreadable path=%r", image_path)
            return image_path

        orig_h, orig_w = image.shape[:2]
        detect_img, scale = _resize_for_detect(image)
        detect_gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        detect_area = float(detect_gray.shape[0] * detect_gray.shape[1])

        cropped = None
        method = None

        quad = _find_page_quad(detect_gray, detect_area)
        if quad is not None:
            # Scale quad back to original image coordinates.
            quad_orig = quad / scale
            try:
                cropped = _four_point_transform(image, quad_orig)
                method = "perspective"
            except Exception as exc:
                logger.debug("perspective transform failed: %s", exc)
                cropped = None

        if cropped is None:
            bbox = _find_largest_bbox(detect_gray, detect_area)
            if bbox is not None:
                x, y, w, h = bbox
                # Scale bbox back to original coordinates.
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
                x, y, w, h = _add_padding(x, y, w, h, orig_w, orig_h)
                cropped = image[y : y + h, x : x + w].copy()
                method = "bbox"

        if cropped is None or cropped.size == 0:
            print("[document_cropper] crop fail: no page region found")
            logger.warning("crop failed: no page region path=%r", image_path)
            return image_path

        crop_h, crop_w = cropped.shape[:2]
        # Reject tiny or near-identical full-frame crops as unsuccessful.
        if crop_w < 40 or crop_h < 40:
            print("[document_cropper] crop fail: crop too small")
            logger.warning("crop failed: too small path=%r", image_path)
            return image_path

        crop_area_ratio = (crop_w * crop_h) / float(max(orig_w * orig_h, 1))
        if crop_area_ratio > 0.95:
            print(
                "[document_cropper] crop fail: crop nearly full frame "
                f"(ratio={crop_area_ratio:.3f})"
            )
            logger.warning(
                "crop failed: nearly full frame ratio=%.3f path=%r",
                crop_area_ratio,
                image_path,
            )
            return image_path

        # Keep full-resolution crop for debug only.
        full_crop_path = _save_crop(cropped, image_path, suffix="crop_full")
        print(f"[document_cropper] debug full crop path: {full_crop_path}")

        # Optimized copy for OCR / classification / BLIP.
        process_img = resize_for_processing(cropped, MAX_PROCESS_WIDTH)
        proc_h, proc_w = process_img.shape[:2]
        out_path = _save_crop(process_img, image_path, suffix="proc")

        print("[document_cropper] crop success")
        print(f"[document_cropper] method: {method}")
        print(f"[document_cropper] cropped image path: {out_path}")
        print(f"[document_cropper] crop size: {crop_w}x{crop_h} -> process {proc_w}x{proc_h}")
        logger.info(
            "crop success method=%s path=%s full=%sx%s process=%sx%s",
            method,
            out_path,
            crop_w,
            crop_h,
            proc_w,
            proc_h,
        )
        return out_path

    except Exception:
        print("[document_cropper] crop fail: unexpected error")
        logger.exception("crop_reading_material failed for path=%r", image_path)
        return image_path


print("Document cropper module loaded successfully")
print("Crop output folder:", CROP_DIR)
