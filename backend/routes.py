"""
RoadGuard AI - Flask REST API Routes
Provides endpoints for dual-head CNN inference, synthetic occlusion simulation,
before/after degradation analysis, model metrics, and system health status.
"""

import os
import json
import traceback
import numpy as np
import tensorflow as tf
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from backend.preprocessing import (
    allowed_file,
    decode_image_bytes,
    decode_base64_image,
    encode_image_to_base64,
    preprocess_for_inference,
    evaluate_safety_policy,
    MAX_FILE_SIZE_BYTES
)
from training.data_loader import GTSRB_CLASSES, get_class_name
from training.occlusion_generator import apply_synthetic_occlusion

api_bp = Blueprint("api", __name__, url_prefix="/api")

# In-memory session tracking for dashboard statistics
STATS = {
    "total_processed": 0,
    "total_occluded": 0,
    "total_clear": 0,
    "recent_history": []
}

_LOADED_MODEL = None
_MODEL_PATH = os.path.join("models", "roadguard_cnn.keras")


def get_model() -> tf.keras.Model:
    """Thread-safe lazy loader for the trained RoadGuard CNN."""
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        if os.path.exists(_MODEL_PATH):
            print(f"[API] Loading RoadGuard model from {_MODEL_PATH}...")
            _LOADED_MODEL = tf.keras.models.load_model(_MODEL_PATH)
            print("[API] RoadGuard model successfully loaded.")
        else:
            return None
    return _LOADED_MODEL


def parse_request_image(req) -> np.ndarray:
    """Extracts and decodes an RGB image from either multipart form-data or JSON base64."""
    if "image" in req.files:
        file = req.files["image"]
        if file.filename == "":
            raise ValueError("No file selected in request")
        if not allowed_file(file.filename):
            raise ValueError("Unsupported file extension. Allowed formats: PNG, JPG, JPEG, WEBP, BMP")
        
        file_bytes = file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB")
        return decode_image_bytes(file_bytes)
        
    elif req.is_json and "image" in req.json:
        b64_data = req.json["image"]
        return decode_base64_image(b64_data)
        
    else:
        raise ValueError("No image provided. Send as multipart form-data field 'image' or JSON payload 'image'")


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Returns application health and model availability status."""
    model_exists = os.path.exists(_MODEL_PATH)
    model_loaded = _LOADED_MODEL is not None
    
    return jsonify({
        "status": "healthy",
        "service": "RoadGuard AI Autonomous Vision Subsystem",
        "model_trained": model_exists,
        "model_loaded": model_loaded,
        "model_path": _MODEL_PATH,
        "total_images_processed": STATS["total_processed"]
    }), 200


@api_bp.route("/classes", methods=["GET"])
def get_classes():
    """Returns all 43 GTSRB traffic sign classes with metadata."""
    formatted_classes = [
        {"id": cid, **meta}
        for cid, meta in GTSRB_CLASSES.items()
    ]
    return jsonify({
        "total_classes": len(formatted_classes),
        "classes": formatted_classes
    }), 200


@api_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns actual calculated model evaluation metrics from results/metrics.json."""
    metrics_path = os.path.join("results", "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({
            "model_trained": False,
            "message": "Model has not been trained yet. No evaluation metrics available."
        }), 200
        
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        
    # Inject current live session count
    metrics_data["session_processed"] = STATS["total_processed"]
    metrics_data["session_occluded"] = STATS["total_occluded"]
    metrics_data["session_clear"] = STATS["total_clear"]
    metrics_data["recent_history"] = STATS["recent_history"][-10:]
    return jsonify(metrics_data), 200


@api_bp.route("/history", methods=["GET"])
def get_training_history():
    """Returns training/validation loss and accuracy curves history."""
    hist_path = os.path.join("results", "history.json")
    if not os.path.exists(hist_path):
        return jsonify({
            "available": False,
            "message": "Training history not found. Train the model first."
        }), 404
        
    with open(hist_path, "r", encoding="utf-8") as f:
        history_data = json.load(f)
    return jsonify({
        "available": True,
        "history": history_data
    }), 200


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Runs dual-output CNN inference on uploaded image:
    1. Sign Classification (43 classes)
    2. Occlusion Detection (0=Clear, 1=Occluded)
    Applies autonomous driving safety policies.
    """
    model = get_model()
    if model is None:
        return jsonify({
            "error": "Model Not Trained",
            "message": "The RoadGuard dual-output CNN model is not trained or loaded. Please train the model first."
        }), 503

    try:
        img_rgb = parse_request_image(request)
    except Exception as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400

    try:
        # Preprocess for CNN
        batch_input = preprocess_for_inference(img_rgb)
        
        # Dual-output inference
        predictions = model.predict(batch_input, verbose=0)
        sign_probs = predictions[0][0]  # shape (43,)
        occ_probs = predictions[1][0]   # shape (2,)
        
        # Traffic Sign Head Results
        predicted_sign_class = int(np.argmax(sign_probs))
        sign_confidence = float(sign_probs[predicted_sign_class])
        sign_name = get_class_name(predicted_sign_class)
        sign_category = GTSRB_CLASSES.get(predicted_sign_class, {}).get("category", "General")
        
        # Top 3 Candidates
        top3_indices = np.argsort(sign_probs)[-3:][::-1]
        top_candidates = [
            {
                "class_id": int(idx),
                "name": get_class_name(int(idx)),
                "confidence": float(sign_probs[idx])
            }
            for idx in top3_indices
        ]
        
        # Occlusion Head Results
        # 0 = Clear, 1 = Occluded
        prob_clear = float(occ_probs[0])
        prob_occluded = float(occ_probs[1])
        is_occluded = bool(prob_occluded >= 0.50)
        occlusion_status = "Occluded" if is_occluded else "Clear"
        occlusion_confidence = prob_occluded if is_occluded else prob_clear
        
        # Module 5: Safety Policy Evaluation
        safety = evaluate_safety_policy(
            sign_confidence=sign_confidence,
            is_occluded=is_occluded,
            occlusion_confidence=occlusion_confidence,
            sign_name=sign_name
        )
        
        # Convert preview to base64
        image_preview_b64 = encode_image_to_base64(img_rgb, format="PNG")
        
        # Update live session statistics
        STATS["total_processed"] += 1
        if is_occluded:
            STATS["total_occluded"] += 1
        else:
            STATS["total_clear"] += 1
            
        history_item = {
            "id": STATS["total_processed"],
            "sign_name": sign_name,
            "sign_class": predicted_sign_class,
            "sign_confidence": round(sign_confidence * 100, 1),
            "occlusion_status": occlusion_status,
            "occlusion_confidence": round(occlusion_confidence * 100, 1),
            "status": safety["status"],
            "severity": safety["severity"]
        }
        STATS["recent_history"].append(history_item)
        if len(STATS["recent_history"]) > 25:
            STATS["recent_history"].pop(0)

        return jsonify({
            "success": True,
            "traffic_sign": {
                "class_id": predicted_sign_class,
                "name": sign_name,
                "category": sign_category,
                "confidence": sign_confidence,
                "confidence_percent": round(sign_confidence * 100, 2),
                "top_candidates": top_candidates
            },
            "occlusion": {
                "is_occluded": is_occluded,
                "status": occlusion_status,
                "confidence": occlusion_confidence,
                "confidence_percent": round(occlusion_confidence * 100, 2),
                "probabilities": {
                    "clear": prob_clear,
                    "occluded": prob_occluded
                }
            },
            "safety": safety,
            "image_preview": image_preview_b64
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Inference Error", "message": str(e)}), 500


@api_bp.route("/simulate-occlusion", methods=["POST"])
def simulate_occlusion():
    """
    Applies synthetic occlusion (leaves, mud, graffiti) with adjustable intensity
    using OpenCV and returns the occluded image.
    """
    try:
        # Determine if JSON or multipart form
        if request.is_json:
            data = request.get_json()
            raw_img = parse_request_image(request)
            occlusion_type = data.get("occlusion_type", "leaves")
            intensity = float(data.get("intensity", 0.5))
        else:
            raw_img = parse_request_image(request)
            occlusion_type = request.form.get("occlusion_type", "leaves")
            intensity = float(request.form.get("intensity", 0.5))
            
        intensity = max(0.1, min(1.0, intensity))
        
        # Apply synthetic occlusion
        occluded_img = apply_synthetic_occlusion(
            image=raw_img,
            occlusion_type=occlusion_type,
            intensity=intensity
        )
        
        orig_b64 = encode_image_to_base64(raw_img, format="PNG")
        occ_b64 = encode_image_to_base64(occluded_img, format="PNG")
        
        return jsonify({
            "success": True,
            "occlusion_type": occlusion_type,
            "intensity": intensity,
            "original_image": orig_b64,
            "occluded_image": occ_b64
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Simulation Error", "message": str(e)}), 400


@api_bp.route("/compare", methods=["POST"])
def compare_analysis():
    """
    Module 4: Before vs After Analysis.
    Runs CNN prediction on original image and simulated occluded image side-by-side.
    Quantifies confidence drops and degradation of autonomous perception.
    """
    model = get_model()
    if model is None:
        return jsonify({
            "error": "Model Not Trained",
            "message": "Please train the model first to perform comparative degradation analysis."
        }), 503

    try:
        if request.is_json:
            data = request.get_json()
            raw_img = parse_request_image(request)
            occlusion_type = data.get("occlusion_type", "leaves")
            intensity = float(data.get("intensity", 0.5))
        else:
            raw_img = parse_request_image(request)
            occlusion_type = request.form.get("occlusion_type", "leaves")
            intensity = float(request.form.get("intensity", 0.5))
            
        intensity = max(0.1, min(1.0, intensity))
        
        # 1. Generate occluded variant
        occluded_img = apply_synthetic_occlusion(
            image=raw_img,
            occlusion_type=occlusion_type,
            intensity=intensity
        )
        
        # 2. Run inference on original
        batch_orig = preprocess_for_inference(raw_img)
        pred_orig = model.predict(batch_orig, verbose=0)
        orig_sign_probs = pred_orig[0][0]
        orig_occ_probs = pred_orig[1][0]
        
        orig_sign_id = int(np.argmax(orig_sign_probs))
        orig_sign_conf = float(orig_sign_probs[orig_sign_id])
        orig_is_occ = bool(orig_occ_probs[1] >= 0.5)
        
        # 3. Run inference on occluded
        batch_occ = preprocess_for_inference(occluded_img)
        pred_occ = model.predict(batch_occ, verbose=0)
        occ_sign_probs = pred_occ[0][0]
        occ_occ_probs = pred_occ[1][0]
        
        new_sign_id = int(np.argmax(occ_sign_probs))
        new_sign_conf = float(occ_sign_probs[new_sign_id])
        new_is_occ = bool(occ_occ_probs[1] >= 0.5)
        
        # Quantify impact
        confidence_delta = round((new_sign_conf - orig_sign_conf) * 100, 2)
        class_swapped = (orig_sign_id != new_sign_id)
        
        orig_safety = evaluate_safety_policy(
            sign_confidence=orig_sign_conf,
            is_occluded=orig_is_occ,
            occlusion_confidence=float(orig_occ_probs[1] if orig_is_occ else orig_occ_probs[0]),
            sign_name=get_class_name(orig_sign_id)
        )
        
        occ_safety = evaluate_safety_policy(
            sign_confidence=new_sign_conf,
            is_occluded=new_is_occ,
            occlusion_confidence=float(occ_occ_probs[1] if new_is_occ else occ_occ_probs[0]),
            sign_name=get_class_name(new_sign_id)
        )
        
        # Explanation text
        if class_swapped:
            explanation = (
                f"Severe occlusion ({occlusion_type.capitalize()} at {int(intensity*100)}% intensity) caused an adversarial perception failure! "
                f"The sign identity flipped from '{get_class_name(orig_sign_id)}' to '{get_class_name(new_sign_id)}'. "
                f"Confidence changed by {confidence_delta}%. Autonomous driving system would require safety fallback."
            )
        elif abs(confidence_delta) > 15:
            explanation = (
                f"Significant perception degradation: Sign identity was retained as '{get_class_name(orig_sign_id)}', "
                f"but classification confidence dropped by {abs(confidence_delta)}% (from {orig_sign_conf*100:.1f}% to {new_sign_conf*100:.1f}%). "
                f"Occlusion head successfully flagged occlusion at {occ_occ_probs[1]*100:.1f}% probability."
            )
        else:
            explanation = (
                f"Minor visual impairment: Model maintains robust classification for '{get_class_name(orig_sign_id)}' "
                f"with {new_sign_conf*100:.1f}% confidence under {int(intensity*100)}% {occlusion_type} occlusion."
            )
            
        return jsonify({
            "success": True,
            "occlusion_type": occlusion_type,
            "intensity": intensity,
            "original": {
                "image": encode_image_to_base64(raw_img),
                "sign_class": orig_sign_id,
                "sign_name": get_class_name(orig_sign_id),
                "sign_confidence": round(orig_sign_conf * 100, 2),
                "is_occluded": orig_is_occ,
                "occlusion_confidence": round(float(orig_occ_probs[1] if orig_is_occ else orig_occ_probs[0]) * 100, 2),
                "safety": orig_safety
            },
            "occluded": {
                "image": encode_image_to_base64(occluded_img),
                "sign_class": new_sign_id,
                "sign_name": get_class_name(new_sign_id),
                "sign_confidence": round(new_sign_conf * 100, 2),
                "is_occluded": new_is_occ,
                "occlusion_confidence": round(float(occ_occ_probs[1] if new_is_occ else occ_occ_probs[0]) * 100, 2),
                "safety": occ_safety
            },
            "comparison": {
                "confidence_delta_percent": confidence_delta,
                "class_swapped": class_swapped,
                "explanation": explanation
            }
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Comparison Error", "message": str(e)}), 400
