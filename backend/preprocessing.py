"""
RoadGuard AI - Backend Image Preprocessing & Safety Logic
Validates uploaded files, resizes and normalizes for CNN inference,
handles Base64 encoding/decoding, and applies autonomous driving safety rules.
"""

import io
import os
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, Optional

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
TARGET_INPUT_SHAPE = (64, 64)
CONFIDENCE_THRESHOLD = 0.60


def allowed_file(filename: str) -> bool:
    """Checks whether the filename has a supported image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into an RGB NumPy uint8 array.
    """
    # Use PIL to safely open and auto-orient image based on EXIF
    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img, dtype=np.uint8)


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decodes a base64 string (with or without data URL header) into an RGB NumPy uint8 array.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    raw_bytes = base64.b64decode(base64_str)
    return decode_image_bytes(raw_bytes)


def encode_image_to_base64(img_rgb: np.ndarray, format: str = "PNG") -> str:
    """
    Encodes an RGB NumPy image array into a base64 Data URL string.
    """
    # Convert RGB to BGR for cv2.imencode
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ext = f".{format.lower()}"
    success, buffer = cv2.imencode(ext, img_bgr)
    if not success:
        raise ValueError("Failed to encode image to base64 buffer")
    b64 = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def preprocess_for_inference(img_rgb: np.ndarray) -> np.ndarray:
    """
    Prepares an RGB image for RoadGuard CNN inference:
    1. Resizes to (64, 64)
    2. Casts to float32
    3. Normalizes pixel values to [0.0, 1.0]
    4. Adds batch dimension: (1, 64, 64, 3)
    """
    h, w = img_rgb.shape[:2]
    if (h, w) != TARGET_INPUT_SHAPE:
        resized = cv2.resize(img_rgb, TARGET_INPUT_SHAPE, interpolation=cv2.INTER_AREA)
    else:
        resized = img_rgb
        
    normalized = resized.astype(np.float32) / 255.0
    batch = np.expand_dims(normalized, axis=0)
    return batch


def evaluate_safety_policy(
    sign_confidence: float,
    is_occluded: bool,
    occlusion_confidence: float,
    sign_name: str
) -> Dict[str, Any]:
    """
    Implements Module 5: Safety Alert System Decision Rules:
    1. If the sign is clear and confidence >= threshold: NORMAL DETECTION.
    2. If occlusion is detected: CAUTION - ROAD SIGN PARTIALLY BLOCKED.
    3. If confidence is below the configured threshold: UNCERTAIN - HUMAN VERIFICATION REQUIRED.
    
    Academic simulation disclaimer: Not a certified real vehicle steering controller.
    """
    if sign_confidence < CONFIDENCE_THRESHOLD:
        status = "Uncertain Detection"
        severity = "danger"
        badge_class = "badge-danger"
        title = "UNCERTAIN – HUMAN VERIFICATION REQUIRED"
        recommendation = (
            f"Classification confidence ({sign_confidence*100:.1f}%) is below safety threshold ({CONFIDENCE_THRESHOLD*100:.0f}%). "
            f"Autonomous vision stack cannot guarantee detection. Requesting manual operator verification or LiDAR/radar fusion."
        )
    elif is_occluded:
        status = "Occluded Sign"
        severity = "warning"
        badge_class = "badge-warning"
        title = "CAUTION – ROAD SIGN PARTIALLY BLOCKED"
        recommendation = (
            f"Sign '{sign_name}' is detected with {occlusion_confidence*100:.1f}% occlusion probability. "
            f"Visual features are partially impaired by environmental interference (leaves/mud/graffiti). Maintain conservative speed."
        )
    else:
        status = "Clear Sign"
        severity = "success"
        badge_class = "badge-success"
        title = "NORMAL DETECTION"
        recommendation = (
            f"Road sign '{sign_name}' is clearly visible with {sign_confidence*100:.1f}% confidence. "
            f"Zero significant occlusion detected ({occlusion_confidence*100:.1f}% confidence clear). Nominal perception parameters."
        )

    return {
        "status": status,
        "severity": severity,
        "badge_class": badge_class,
        "alert_title": title,
        "recommendation": recommendation,
        "disclaimer": "Academic simulation for research purposes. Not certified for vehicle actuator control."
    }
