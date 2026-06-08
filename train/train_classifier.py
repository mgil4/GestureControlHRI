#!/usr/bin/env python3
"""
Train Gesture Classifier
------------------------
Trains a classifier using gesture landmark data stored in gesture_data.csv.

Features:
  - Train / Validation / Test split (stratified)
  - Accuracy, Precision, Recall, F1 metrics
  - Confusion matrix (error-focused, zoomed on mistakes)
  - Per-label top confusions analysis
  - Equivalent-sign aware evaluation (b=4, d=1, f=9, o=0, v=2, w=6)
    applied consistently to: reports, error matrix, confusion analysis
  - Training + Validation loss curve
  - External CSV test evaluation
  - Modular structure (logic split out of main)

Install:
    pip install numpy scikit-learn matplotlib pandas

Run:
    python train_classifier.py
"""

import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_CSV        = "gesture_data.csv"
EXTRA_DATA_CSV  = r"C:\Users\maria\Downloads\extra_test_asl_data.csv"
MODEL_SAVE_PATH = "gesture_classifier.pkl"

TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
RANDOM_SEED = 42

# Visually equivalent sign pairs (lowercase).
# Predicting one when the true label is the other is NOT an error.
EQUIVALENT_PAIRS = [
    ("b", "4"),
    ("d", "1"),
    ("f", "9"),
    ("o", "0"),
    ("v", "2"),
    ("w", "6"),
]

def build_equivalents_map(pairs):
    equiv = {}
    for a, b in pairs:
        equiv.setdefault(a, set()).add(b)
        equiv.setdefault(b, set()).add(a)
    return equiv

EQUIVALENTS = build_equivalents_map(EQUIVALENT_PAIRS)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_data(csv_path):
    X, y = [], []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    return np.array(X, dtype=np.float32), np.array(y)


def split_data(X, y_enc, test_size, val_size, random_seed):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,
        stratify=y_enc,
        random_state=random_seed,
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        stratify=y_trainval,
        random_state=random_seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_distribution(name, labels):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{name} distribution:")
    print("-" * 40)
    for u, c in zip(unique, counts):
        print(f"  {u:15s}: {c}")


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model(random_seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=64,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_seed,
            verbose=True,
        ))
    ])


# ─────────────────────────────────────────────
# EQUIVALENT-AWARE HELPERS
# ─────────────────────────────────────────────

def apply_equivalents_to_predictions(y_true_labels, y_pred_labels, equivalents):
    """
    If a prediction is a visual equivalent of the true label, replace it with
    the true label so it registers as correct in every downstream metric.

    Example: true='b', pred='4'  ->  corrected pred='b'
    """
    corrected = []
    for t, p in zip(y_true_labels, y_pred_labels):
        if p in equivalents.get(t, set()):
            corrected.append(t)
        else:
            corrected.append(p)
    return np.array(corrected)


def adjusted_accuracy(y_true_labels, y_pred_labels, equivalents):
    """
    Scalar accuracy where equivalent-pair predictions count as correct.
    """
    correct = 0
    for t, p in zip(y_true_labels, y_pred_labels):
        if t == p or p in equivalents.get(t, set()):
            correct += 1
    return correct / len(y_true_labels)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(clf, le, X, y_true_enc, split_name, out_prefix):
    """
    Raw evaluation (no equivalent remapping).
    Returns raw predicted encoded labels and raw accuracy.
    The caller applies equivalent remapping before reports / plots.
    """
    y_pred_enc = clf.predict(X)
    acc = accuracy_score(y_true_enc, y_pred_enc)
    print(f"\n{split_name} Accuracy (raw): {acc:.4f}")
    return y_pred_enc, acc


def build_report(y_true_enc, y_pred_enc_adj, le, only_present=False):
    """
    Build a classification_report string using equivalent-adjusted predictions.
    Set only_present=True when the label set may be incomplete (extra test).
    """
    if only_present:
        present = np.unique(np.concatenate([y_true_enc, y_pred_enc_adj]))
        return classification_report(
            y_true_enc, y_pred_enc_adj,
            labels=present,
            target_names=le.classes_[present],
            digits=4,
        )
    return classification_report(
        y_true_enc, y_pred_enc_adj,
        target_names=le.classes_,
        digits=4,
    )


# ─────────────────────────────────────────────
# CONFUSION ANALYSIS
# ─────────────────────────────────────────────

def top_confusions_per_label(y_true_labels, y_pred_labels_adj, le, top_n=3):
    """
    For each label, show which labels it is most often confused with.
    Receives already-equivalent-adjusted predictions, so equivalent pairs
    never appear here as errors.
    """
    classes = list(le.classes_)
    label_to_idx = {c: i for i, c in enumerate(classes)}

    present = set(y_true_labels) | set(y_pred_labels_adj)
    present_classes = [c for c in classes if c in present]
    present_idx = [label_to_idx[c] for c in present_classes]

    y_true_enc = [label_to_idx[t] for t in y_true_labels]
    y_pred_enc = [label_to_idx[p] for p in y_pred_labels_adj]

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=present_idx)

    print("\nTop confusions per label (true -> predicted as, after equiv remapping):")
    print("=" * 60)

    confusion_data = {}
    for i, label in enumerate(present_classes):
        row = cm[i].copy()
        row[i] = 0  # zero out correct
        total_errors = row.sum()
        if total_errors == 0:
            continue
        top_idx = np.argsort(row)[::-1][:top_n]
        top = [(present_classes[j], int(row[j])) for j in top_idx if row[j] > 0]
        confusion_data[label] = {"total_errors": int(total_errors), "top": top}
        top_str = ", ".join([f"'{p}' x{c}" for p, c in top])
        print(f"  {label:4s} ({total_errors:3d} errors) -> {top_str}")

    return confusion_data


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_loss_curve(mlp, save_path="training_loss.png"):
    """
    Training loss + validation loss proxy (1 - val_accuracy) on the same axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mlp.loss_curve_, label="Train Loss", color="steelblue", linewidth=2)

    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        val_loss_proxy = [1 - s for s in mlp.validation_scores_]
        ax.plot(val_loss_proxy, label="Val Loss (1 - acc)",
                color="tomato", linewidth=2, linestyle="--")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_matrix(y_true_labels, y_pred_labels_adj, le,
                      save_path="confusion_matrix.png", title_suffix=""):
    """
    Error-focused matrix built from equivalent-adjusted predictions:
      - Diagonal zeroed out (correct predictions invisible).
      - Rows/cols with zero errors removed.
      - Cell counts annotated directly on the heatmap.
    """
    classes = list(le.classes_)
    present = sorted(set(y_true_labels) | set(y_pred_labels_adj),
                     key=lambda c: classes.index(c) if c in classes else 999)
    idx_map = {c: i for i, c in enumerate(present)}

    y_true_enc = [idx_map[t] for t in y_true_labels]
    y_pred_enc = [idx_map[p] for p in y_pred_labels_adj]
    n = len(present)
    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=np.arange(n))

    err = cm.copy().astype(float)
    np.fill_diagonal(err, 0)

    has_error = (err.sum(axis=1) > 0) | (err.sum(axis=0) > 0)
    filtered = [c for c, keep in zip(present, has_error) if keep]
    err_f = err[np.ix_(has_error, has_error)]

    if err_f.size == 0:
        print(f"No errors found -- skipping error matrix ({save_path}).")
        return

    fig_size = max(8, len(filtered) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(err_f, cmap=plt.cm.Reds, aspect="auto")

    ax.set_xticks(np.arange(len(filtered)))
    ax.set_yticks(np.arange(len(filtered)))
    ax.set_xticklabels(filtered, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(filtered, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    title = "Error Matrix -- equiv pairs treated as correct"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=12)

    max_val = err_f.max() or 1
    for i in range(len(filtered)):
        for j in range(len(filtered)):
            v = int(err_f[i, j])
            if v > 0:
                color = "white" if v > max_val * 0.6 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Number of errors")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# EXTRA TEST
# ─────────────────────────────────────────────

def evaluate_extra_test(clf, le, csv_path):
    """Load an external CSV and evaluate the trained model on it."""
    print(f"\n{'='*60}")
    print(f"EXTRA TEST EVALUATION: {csv_path}")
    print(f"{'='*60}")

    X_ext, y_ext_raw = load_data(csv_path)

    known_mask = np.isin(y_ext_raw, le.classes_)
    unknown = set(y_ext_raw[~known_mask])
    if unknown:
        print(f"  Warning: skipping {(~known_mask).sum()} samples "
              f"with unknown labels: {unknown}")
    X_ext     = X_ext[known_mask]
    y_ext_raw = y_ext_raw[known_mask]

    if len(X_ext) == 0:
        print("  No valid samples found in extra test file.")
        return

    y_ext_enc      = le.transform(y_ext_raw)
    y_ext_pred_enc = clf.predict(X_ext)
    y_ext_pred_raw = le.inverse_transform(y_ext_pred_enc)

    # Raw accuracy
    raw_acc = accuracy_score(y_ext_enc, y_ext_pred_enc)

    # Apply equivalent remapping
    y_ext_pred_adj     = apply_equivalents_to_predictions(y_ext_raw, y_ext_pred_raw, EQUIVALENTS)
    y_ext_pred_enc_adj = le.transform(y_ext_pred_adj)
    adj_acc            = adjusted_accuracy(y_ext_raw, y_ext_pred_raw, EQUIVALENTS)

    # Report built on adjusted predictions
    report = build_report(y_ext_enc, y_ext_pred_enc_adj, le, only_present=True)

    print(f"\nExtra Test Raw Accuracy      : {raw_acc:.4f}")
    print(f"Extra Test Adjusted Accuracy : {adj_acc:.4f}  (equiv. pairs not counted as errors)")
    print("\nClassification Report (equiv-adjusted):")
    print(report)

    with open("extra_test_report.txt", "w") as f:
        f.write(f"Extra Test Raw Accuracy: {raw_acc:.4f}\n")
        f.write(f"Extra Test Adjusted Accuracy: {adj_acc:.4f}\n\n")
        f.write(report)

    top_confusions_per_label(y_ext_raw, y_ext_pred_adj, le)
    plot_error_matrix(y_ext_raw, y_ext_pred_adj, le,
                      save_path="extra_test_error_matrix.png",
                      title_suffix="extra test")

    print("\nSaved: extra_test_report.txt")
    print("Saved: extra_test_error_matrix.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Load & split
    print("Loading dataset...")
    X, y = load_data(DATA_CSV)
    print(f"Total samples : {len(X)}")
    print(f"Total classes : {len(set(y))}")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y_enc, TEST_SIZE, VAL_SIZE, RANDOM_SEED
    )

    print_distribution("TRAIN",      le.inverse_transform(y_train))
    print_distribution("VALIDATION", le.inverse_transform(y_val))
    print_distribution("TEST",       le.inverse_transform(y_test))

    # Train
    clf = build_model(RANDOM_SEED)
    print("\nTraining classifier...")
    clf.fit(X_train, y_train)

    # Raw predictions
    y_val_pred_enc,  val_acc  = evaluate(clf, le, X_val,  y_val,  "Validation", "validation")
    y_test_pred_enc, test_acc = evaluate(clf, le, X_test, y_test, "Test",       "test")

    # Convert to label strings
    y_val_labels     = le.inverse_transform(y_val)
    y_test_labels    = le.inverse_transform(y_test)
    val_pred_labels  = le.inverse_transform(y_val_pred_enc)
    test_pred_labels = le.inverse_transform(y_test_pred_enc)

    # Apply equivalent remapping
    val_pred_adj      = apply_equivalents_to_predictions(y_val_labels,  val_pred_labels,  EQUIVALENTS)
    test_pred_adj     = apply_equivalents_to_predictions(y_test_labels, test_pred_labels, EQUIVALENTS)
    val_pred_enc_adj  = le.transform(val_pred_adj)
    test_pred_enc_adj = le.transform(test_pred_adj)

    val_adj  = adjusted_accuracy(y_val_labels,  val_pred_labels,  EQUIVALENTS)
    test_adj = adjusted_accuracy(y_test_labels, test_pred_labels, EQUIVALENTS)

    print(f"\nValidation Adjusted Accuracy : {val_adj:.4f}  (equiv. pairs not counted as errors)")
    print(f"Test Adjusted Accuracy : {test_adj:.4f}  (equiv. pairs not counted as errors)")

    # Reports (equiv-adjusted)
    val_report  = build_report(y_val,  val_pred_enc_adj,  le)
    test_report = build_report(y_test, test_pred_enc_adj, le)

    print("\nValidation Classification Report (equiv-adjusted):")
    print(val_report)
    print("\nTest Classification Report (equiv-adjusted):")
    print(test_report)

    with open("validation_report.txt", "w") as f:
        f.write(f"Validation Raw Accuracy: {val_acc:.4f}\n")
        f.write(f"Validation Adjusted Accuracy: {val_adj:.4f}\n\n")
        f.write(val_report)

    with open("test_report.txt", "w") as f:
        f.write(f"Test Raw Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Adjusted Accuracy: {test_adj:.4f}\n\n")
        f.write(test_report)

    # Confusion analysis (equiv-adjusted)
    print("\n--- VALIDATION ---")
    top_confusions_per_label(y_val_labels,  val_pred_adj,  le)
    print("\n--- TEST ---")
    top_confusions_per_label(y_test_labels, test_pred_adj, le)

    # Plots
    mlp = clf.named_steps["mlp"]
    plot_loss_curve(mlp, "training_loss.png")
    plot_error_matrix(y_test_labels, test_pred_adj, le,
                      save_path="confusion_matrix.png", title_suffix="test set")

    # Save model
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump((clf, le), f)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    # Summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")
    print(f"  Test samples       : {len(X_test)}")
    print(f"  Validation Accuracy: {val_acc:.4f}  (adj: {val_adj:.4f})")
    print(f"  Test Accuracy      : {test_acc:.4f}  (adj: {test_adj:.4f})")
    print("\nSaved files:")
    for fname in [
        "gesture_classifier.pkl", "validation_report.txt",
        "test_report.txt", "confusion_matrix.png", "training_loss.png"
    ]:
        print(f"  - {fname}")

    # Extra test
    evaluate_extra_test(clf, le, EXTRA_DATA_CSV)


if __name__ == "__main__":
    main()

