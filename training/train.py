"""
RoadGuard AI - Dual-Output CNN Training Pipeline
Jointly trains:
1. Traffic Sign Classification (43 classes)
2. Binary Occlusion Detection (Clear vs Occluded)
Saves best model, training history curves, and triggers evaluation on test split.
"""

import os
import sys
import argparse
import random
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure temp matplotlib config directory
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_roadguard")

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TerminateOnNaN
)

from training.data_loader import prepare_dual_output_splits
from training.cnn_model import build_roadguard_model
from training.evaluate import evaluate_roadguard_model, plot_training_history


def set_reproducible_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Reproducibility seeds set to {seed}")


def run_training(
    dataset_dir: str = "dataset",
    models_dir: str = "models",
    results_dir: str = "results",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42
):
    set_reproducible_seeds(seed)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "roadguard_cnn.keras")
    
    print("\n" + "="*60)
    print("  ROADGUARD AI - PREPARING DATASET SPLITS")
    print("="*60)
    
    splits = prepare_dual_output_splits(
        dataset_dir=dataset_dir,
        target_size=(64, 64),
        test_size=0.15,
        val_size=0.15,
        occlusion_ratio=0.50,
        seed=seed
    )
    
    X_train, y_train_sign, y_train_occ = splits["train"]
    X_val, y_val_sign, y_val_occ = splits["val"]
    X_test, y_test_sign, y_test_occ = splits["test"]
    
    print(f"\nFinal Dataset Shapes:")
    print(f"  Training:   X={X_train.shape}, Signs={y_train_sign.shape}, Occlusion={y_train_occ.shape}")
    print(f"  Validation: X={X_val.shape}, Signs={y_val_sign.shape}, Occlusion={y_val_occ.shape}")
    print(f"  Testing:    X={X_test.shape}, Signs={y_test_sign.shape}, Occlusion={y_test_occ.shape}")
    
    print("\n" + "="*60)
    print("  BUILDING DUAL-OUTPUT CNN ARCHITECTURE")
    print("="*60)
    
    model = build_roadguard_model(
        input_shape=(64, 64, 3),
        num_sign_classes=43,
        num_occlusion_classes=2,
        dropout_rate=0.20,
        learning_rate=learning_rate
    )
    model.summary()
    
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_sign_output_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_sign_output_accuracy",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        TerminateOnNaN()
    ]
    
    print("\n" + "="*60)
    print(f"  COMMENCING JOINT DUAL-HEAD TRAINING ({epochs} MAX EPOCHS)")
    print("="*60)
    
    history = model.fit(
        x=X_train,
        y={
            "sign_output": y_train_sign,
            "occlusion_output": y_train_occ
        },
        validation_data=(
            X_val,
            {
                "sign_output": y_val_sign,
                "occlusion_output": y_val_occ
            }
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    print(f"\nTraining completed. Saving final model checkpoint to {model_save_path}...")
    model.save(model_save_path)
    
    print("\nGenerating training history visualization...")
    plot_training_history(history.history, results_dir=results_dir)
    
    print("\nEvaluating best model on held-out test split...")
    metrics = evaluate_roadguard_model(
        model=model,
        X_test=X_test,
        y_test_sign=y_test_sign,
        y_test_occ=y_test_occ,
        results_dir=results_dir
    )
    
    print("\n" + "="*60)
    print("  ROADGUARD AI TRAINING & EVALUATION SUCCESSFULLY FINISHED")
    print("="*60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoadGuard AI Dual-Output CNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    run_training(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed
    )
