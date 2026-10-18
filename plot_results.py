"""Draw the detection matrix and a compact summary of model scores."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def plot_detection_matrix(results_dir, figures_dir):
    matrix = pd.read_csv(results_dir / "detection_matrix.csv", index_col=0)
    columns = ["Source isolation", "Exact duplicate", "Generated provenance"]
    values = matrix[columns]
    levels = {"PASS": 0, "DETECTED": 1, "MISSED": 2, "FALSE POSITIVE": 3}
    codes = np.array([[levels[value] for value in row] for row in values.to_numpy()])

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    ax.imshow(codes, cmap=ListedColormap(["#e8edf1", "#b8dfcc", "#f5cd9e", "#e9b1b1"]),
              vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(columns)), ["Source isolation", "Exact duplicate", "Generated\nprovenance"])
    ax.set_yticks(range(len(values)), values.index)
    ax.tick_params(length=0)
    for row in range(len(values)):
        for col in range(len(columns)):
            ax.text(col, row, values.iloc[row, col], ha="center", va="center", fontsize=11)
    ax.set_title("Leakage detection: six controlled scenarios", pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(figures_dir / "detection_matrix.png", dpi=180)
    plt.close(fig)


def plot_model_performance(results_dir, figures_dir):
    metrics = pd.read_csv(results_dir / "pilot_model_performance.csv", index_col=0)
    columns = ["Accuracy", "F1", "Average precision (AP)"]
    titles = ["Accuracy", "F1, class 1", "Average precision (AP)"]
    positions = np.arange(len(metrics))
    colors = ["#b46f3c" if name.startswith("Buggy") else "#357d8a" for name in metrics.index]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.3), sharey=True, constrained_layout=True)
    for ax, column, title in zip(axes, columns, titles):
        scores = metrics[column].to_numpy()
        ax.barh(positions, scores, color=colors)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks(positions, metrics.index)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for position, score in zip(positions, scores):
            ax.text(score + 0.02, position, f"{score:.3f}", va="center", fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Same original Test records; faulty-model scores are contaminated", fontsize=12)
    fig.savefig(figures_dir / "model_performance.png", dpi=180)
    plt.close(fig)


def main():
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_detection_matrix(results_dir, figures_dir)
    plot_model_performance(results_dir, figures_dir)
    print("Saved figures/detection_matrix.png and figures/model_performance.png")


if __name__ == "__main__":
    main()
