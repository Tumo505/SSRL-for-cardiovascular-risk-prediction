from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def set_publication_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "figure.figsize": (8, 6),
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    ax=None,
    color: str = "b",
) -> tuple:
    """Plot ROC curve for binary or averaged multi-class."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        y_prob_flat = y_prob.mean(axis=1)
    else:
        y_prob_flat = y_prob.squeeze()

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true_flat = y_true.mean(axis=1)
    else:
        y_true_flat = y_true.squeeze()

    fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.axline((0, 0), slope=1, color="k", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic Curve", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    return fig, ax, roc_auc


def plot_label_efficiency(
    label_fractions: list,
    supervised_auroc: list,
    ssl_auroc: list,
    supervised_std: list | None = None,
    ssl_std: list | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot label efficiency comparison: AUROC vs label fraction."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    fractions_pct = [f * 100 for f in label_fractions]

    if supervised_std is not None:
        ax.fill_between(fractions_pct, np.array(supervised_auroc) - np.array(supervised_std),
                         np.array(supervised_auroc) + np.array(supervised_std), alpha=0.2, color="C0")
    ax.plot(fractions_pct, supervised_auroc, "o-", color="C0", linewidth=2.5, markersize=8, label="Supervised Baseline")

    if ssl_std is not None:
        ax.fill_between(fractions_pct, np.array(ssl_auroc) - np.array(ssl_std),
                         np.array(ssl_auroc) + np.array(ssl_std), alpha=0.2, color="C1")
    ax.plot(fractions_pct, ssl_auroc, "s-", color="C1", linewidth=2.5, markersize=8, label="SSL + Fine-tune")

    ax.set_xlabel("Fraction of Labelled Training Data (%)", fontsize=12)
    ax.set_ylabel("AUROC (macro average)", fontsize=12)
    ax.set_title("Label Efficiency: Self-Supervised vs Supervised Learning", fontsize=13)
    ax.set_xscale("log")
    ax.set_xlim([0.5, 150])
    ax.set_ylim([0.5, 1.0])
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved label efficiency plot to: {output_path}")
    else:
        plt.show()


def plot_robustness_comparison(
    model_names: list,
    clean_scores: list,
    noise_scores: list,
    mask_scores: list,
    metric_name: str = "AUROC",
    output_path: Path | None = None,
) -> None:
    """Plot robustness under noise and masking."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(model_names))
    width = 0.25

    ax.bar(x - width, clean_scores, width, label="Clean", color="C0", alpha=0.8)
    ax.bar(x, noise_scores, width, label="Noisy (+10% std)", color="C1", alpha=0.8)
    ax.bar(x + width, mask_scores, width, label="Masked (20%)", color="C2", alpha=0.8)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Model Robustness Under Real-World Corruptions ({metric_name})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis="y")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved robustness plot to: {output_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot confusion matrix for multi-class classification."""
    from sklearn.metrics import confusion_matrix

    set_publication_style()

    if y_true.ndim == 2:
        y_true_classes = np.argmax(y_true, axis=1)
    else:
        y_true_classes = y_true

    if y_pred.ndim == 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(8, n_classes), max(8, n_classes)))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to: {output_path}")
    else:
        plt.show()


def plot_signal_examples(
    signals: list[np.ndarray],
    labels: list[str],
    titles: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot example ECG signals."""
    set_publication_style()

    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(10, 3 * n_signals))

    if n_signals == 1:
        axes = [axes]

    for i, (sig, label) in enumerate(zip(signals, labels)):
        ax = axes[i]
        time = np.arange(len(sig)) / 100.0
        ax.plot(time, sig, color="C0", linewidth=1.5)
        ax.set_ylabel("Amplitude (mV)", fontsize=11)
        title = titles[i] if titles else f"Signal {i + 1}"
        ax.set_title(f"{title} ({label})", fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved signal examples to: {output_path}")
    else:
        plt.show()
