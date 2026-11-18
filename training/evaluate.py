"""
RoadGuard AI - Comprehensive Model Evaluation & Metrics Generator
Evaluates the dual-output CNN on the held-out test split.
Calculates Sign Classification Accuracy, Macro/Weighted F1, Occlusion Precision,
Recall, F1-Score, generates Confusion Matrix and Training History visualisations.
"""

import os
import sys
import json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure temp matplotlib config directory
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_roadguard")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import tensorflow as tf

# Import local modules
from training.data_loader import GTSRB_CLASSES, get_class_name


def evaluate_roadguard_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test_sign: np.ndarray,
    y_test_occ: np.ndarray,
    results_dir: str = "results"
) -> dict:
    """
    Evaluates the model on test split and exports all metrics and plots.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("  ROADGUARD AI - EVALUATION ON TEST SET")
    print("="*60)
    
    # Run model inference on test set
    preds = model.predict(X_test, batch_size=32, verbose=1)
    preds_sign = preds[0]  # shape: (N, 43)
    preds_occ = preds[1]   # shape: (N, 2)
    
    y_pred_sign = np.argmax(preds_sign, axis=1)
    y_pred_occ = np.argmax(preds_occ, axis=1)
    
    # ------------------ Head 1: Traffic Sign Classification ------------------ #
    sign_acc = float(accuracy_score(y_test_sign, y_pred_sign))
    sign_f1_macro = float(f1_score(y_test_sign, y_pred_sign, average="macro", zero_division=0))
    sign_f1_weighted = float(f1_score(y_test_sign, y_pred_sign, average="weighted", zero_division=0))
    
    # ------------------ Head 2: Occlusion Detection ------------------ #
    occ_acc = float(accuracy_score(y_test_occ, y_pred_occ))
    occ_precision = float(precision_score(y_test_occ, y_pred_occ, pos_label=1, zero_division=0))
    occ_recall = float(recall_score(y_test_occ, y_pred_occ, pos_label=1, zero_division=0))
    occ_f1 = float(f1_score(y_test_occ, y_pred_occ, pos_label=1, zero_division=0))
    
    # Confusion Matrix for Occlusion (2x2: Clear vs Occluded)
    cm_occ = confusion_matrix(y_test_occ, y_pred_occ, labels=[0, 1])
    tn, fp, fn, tp = cm_occ.ravel() if cm_occ.size == 4 else (0, 0, 0, 0)
    
    # Total occluded detected
    total_occluded_detected = int(np.sum(y_pred_occ == 1))
    total_clear_detected = int(np.sum(y_pred_occ == 0))
    
    print(f"\n[METRICS SUMMARY]")
    print(f"Total Test Samples:            {len(X_test)}")
    print(f"Sign Classification Accuracy:  {sign_acc * 100:.2f}%")
    print(f"Sign Macro F1-Score:           {sign_f1_macro:.4f}")
    print(f"Occlusion Detection Accuracy:  {occ_acc * 100:.2f}%")
    print(f"Occlusion Detection Precision: {occ_precision * 100:.2f}%")
    print(f"Occlusion Detection Recall:    {occ_recall * 100:.2f}%")
    print(f"Occlusion Detection F1-Score:  {occ_f1 * 100:.2f}%")
    print(f"Confusion Matrix (Occlusion):  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    # ------------------ Generate Confusion Matrix Heatmap ------------------ #
    plt.figure(figsize=(7, 6), facecolor="#071426")
    ax = plt.gca()
    ax.set_facecolor("#0D1B2A")
    
    sns.heatmap(
        cm_occ,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Clear (0)", "Occluded (1)"],
        yticklabels=["Clear (0)", "Occluded (1)"],
        cbar=True,
        annot_kws={"size": 14, "weight": "bold", "color": "white"}
    )
    plt.title("Occlusion Detection Confusion Matrix\n(RoadGuard AI Dual-Head CNN)", color="#00D9FF", fontsize=13, pad=15)
    plt.xlabel("Predicted Class", color="#E2E8F0", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth Class", color="#E2E8F0", fontsize=11, labelpad=10)
    ax.tick_params(colors="#94A3B8")
    
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200, facecolor="#071426")
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")
    
    # ------------------ Generate Classification Report Text ------------------ #
    target_names_occ = ["Clear", "Occluded"]
    report_occ = classification_report(
        y_test_occ, y_pred_occ,
        target_names=target_names_occ,
        zero_division=0
    )
    
    # Unique classes present in test set for signs
    unique_sign_classes = sorted(list(set(y_test_sign)))
    sign_names = [f"Class {c:02d}: {get_class_name(c)[:25]}" for c in unique_sign_classes]
    report_sign = classification_report(
        y_test_sign, y_pred_sign,
        labels=unique_sign_classes,
        target_names=sign_names,
        zero_division=0
    )
    
    full_report = (
        "="*70 + "\n"
        "           ROADGUARD AI - MODEL PERFORMANCE REPORT\n"
        "="*70 + "\n\n"
        "1. DUAL-OUTPUT SUMMARY\n"
        "----------------------------------------------------------------------\n"
        f"Traffic Sign Classification Accuracy: {sign_acc * 100:.2f}%\n"
        f"Traffic Sign Macro F1-Score:          {sign_f1_macro:.4f}\n"
        f"Traffic Sign Weighted F1-Score:       {sign_f1_weighted:.4f}\n\n"
        f"Occlusion Detection Accuracy:         {occ_acc * 100:.2f}%\n"
        f"Occlusion Detection Precision:        {occ_precision * 100:.2f}%\n"
        f"Occlusion Detection Recall:           {occ_recall * 100:.2f}%\n"
        f"Occlusion Detection F1-Score:         {occ_f1 * 100:.2f}%\n"
        f"Occlusion Confusion Matrix (TP/FP/TN/FN): TP={tp}, FP={fp}, TN={tn}, FN={fn}\n\n"
        "2. OCCLUSION CLASSIFICATION REPORT\n"
        "----------------------------------------------------------------------\n"
        f"{report_occ}\n"
        "3. TRAFFIC SIGN CLASSIFICATION REPORT (SUBSET OF TEST CLASSES)\n"
        "----------------------------------------------------------------------\n"
        f"{report_sign}\n"
        "="*70 + "\n"
    )
    
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"Saved classification report: {report_path}")
    
    # ------------------ Save Metrics JSON for Dashboard ------------------ #
    metrics_data = {
        "model_trained": True,
        "total_test_samples": int(len(X_test)),
        "sign_accuracy": round(sign_acc, 4),
        "sign_f1_macro": round(sign_f1_macro, 4),
        "sign_f1_weighted": round(sign_f1_weighted, 4),
        "occlusion_accuracy": round(occ_acc, 4),
        "occlusion_precision": round(occ_precision, 4),
        "occlusion_recall": round(occ_recall, 4),
        "occlusion_f1": round(occ_f1, 4),
        "total_occluded_detected": total_occluded_detected,
        "total_clear_detected": total_clear_detected,
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn)
        },
        "evaluation_timestamp": str(np.datetime64("now"))
    }
    
    json_path = os.path.join(results_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics JSON: {json_path}")
    
    return metrics_data


def plot_training_history(history_dict: dict, results_dir: str = "results") -> str:
    """
    Plots Loss and Accuracy curves for both heads (Sign & Occlusion) across epochs.
    Saves the figure to results/training_history.png.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    epochs = range(1, len(history_dict.get("loss", [])) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#071426")
    for ax in axes:
        ax.set_facecolor("#0D1B2A")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94A3B8")
        ax.tick_params(colors="#94A3B8")
    
    # 1. Total & Head Losses
    ax_loss = axes[0]
    ax_loss.plot(epochs, history_dict.get("loss", []), label="Train Total Loss", color="#00D9FF", lw=2)
    if "val_loss" in history_dict:
        ax_loss.plot(epochs, history_dict["val_loss"], label="Val Total Loss", color="#2563EB", lw=2, linestyle="--")
    if "sign_output_loss" in history_dict:
        ax_loss.plot(epochs, history_dict["sign_output_loss"], label="Sign Loss", color="#F59E0B", lw=1.5, alpha=0.8)
    if "occlusion_output_loss" in history_dict:
        ax_loss.plot(epochs, history_dict["occlusion_output_loss"], label="Occlusion Loss", color="#EF4444", lw=1.5, alpha=0.8)
        
    ax_loss.set_title("Training & Validation Loss Curves", color="#00D9FF", fontsize=13, pad=12)
    ax_loss.set_xlabel("Epoch", color="#E2E8F0", fontsize=11)
    ax_loss.set_ylabel("Loss", color="#E2E8F0", fontsize=11)
    ax_loss.legend(facecolor="#071426", edgecolor="#2563EB", labelcolor="#F8FAFC")
    
    # 2. Accuracies
    ax_acc = axes[1]
    if "sign_output_accuracy" in history_dict:
        ax_acc.plot(epochs, history_dict["sign_output_accuracy"], label="Train Sign Acc", color="#22C55E", lw=2)
    if "val_sign_output_accuracy" in history_dict:
        ax_acc.plot(epochs, history_dict["val_sign_output_accuracy"], label="Val Sign Acc", color="#10B981", lw=2, linestyle="--")
    if "occlusion_output_accuracy" in history_dict:
        ax_acc.plot(epochs, history_dict["occlusion_output_accuracy"], label="Train Occlusion Acc", color="#00D9FF", lw=2)
    if "val_occlusion_output_accuracy" in history_dict:
        ax_acc.plot(epochs, history_dict["val_occlusion_output_accuracy"], label="Val Occlusion Acc", color="#38BDF8", lw=2, linestyle="--")
        
    ax_acc.set_title("Dual-Head Classification Accuracy", color="#00D9FF", fontsize=13, pad=12)
    ax_acc.set_xlabel("Epoch", color="#E2E8F0", fontsize=11)
    ax_acc.set_ylabel("Accuracy", color="#E2E8F0", fontsize=11)
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.legend(facecolor="#071426", edgecolor="#2563EB", labelcolor="#F8FAFC")
    
    plt.tight_layout()
    out_file = os.path.join(results_dir, "training_history.png")
    plt.savefig(out_file, dpi=200, facecolor="#071426")
    plt.close()
    print(f"Saved training history curves: {out_file}")
    
    # Also save raw history numbers in JSON for Chart.js rendering
    hist_json_path = os.path.join(results_dir, "history.json")
    clean_history = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    with open(hist_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_history, f, indent=2)
        
    return out_file
