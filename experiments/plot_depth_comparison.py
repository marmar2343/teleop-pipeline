"""
plot_depth_comparison.py

Plots results from compare_depth_methods.py.
Input: rezultati_dubina.json
Output: images/depth_*.pdf/.png + depth_comparison.tex
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {
    "realsense": "RealSense (reference)",
    "yolo": "YOLO26-depth",
    "depthany": "Depth Anything V2",
    "mp_size": "Apparent hand size",
}
COLORS = {"realsense": "#2e7d32", "yolo": "#1f77b4",
          "depthany": "#9467bd", "mp_size": "#d62728"}

os.makedirs("images", exist_ok=True)


def get_name(m):
    return NAMES.get(m, m)


# ---------- Figure 1: estimated vs. actual distance ----------
def plot_estimated_vs_actual(results, fname="images/depth_estimated"):
    labels = np.array(results["labels_cm"]) / 100.0
    fig, ax = plt.subplots(figsize=(7.5, 6))

    lo, hi = labels.min() - 0.05, labels.max() + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal", zorder=1)

    for m, data in results["methods"].items():
        pred = np.array(data["pred"])
        ok = np.isfinite(pred)
        # Small horizontal offset prevents points from overlapping
        jitter = (list(results["methods"]).index(m) - 1) * 0.004
        ax.scatter(labels[ok] + jitter, pred[ok], s=14, alpha=0.5,
                   color=COLORS.get(m, "gray"), label=get_name(m), zorder=2)

    ax.set_xlabel("Actual distance (ruler) [m]")
    ax.set_ylabel("Estimated distance [m]")
    ax.set_title("Depth estimation by method")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fname}.pdf")


# ---------- Figure 2: bias and noise by distance ----------
def plot_by_distance(results, fname="images/depth_by_distance"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for m, data in results["methods"].items():
        pd = data["vs_ruler"]["per_distance"]
        ds = sorted(int(k) for k in pd)
        bias = [pd[str(d)]["bias_mm"] if str(d) in pd else pd[d]["bias_mm"] for d in ds]
        std = [pd[str(d)]["std_mm"] if str(d) in pd else pd[d]["std_mm"] for d in ds]

        axes[0].plot(ds, bias, "o-", color=COLORS.get(m, "gray"), label=get_name(m))
        axes[1].plot(ds, std, "o-", color=COLORS.get(m, "gray"), label=get_name(m))

    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Distance [cm]")
    axes[0].set_ylabel("Systematic error [mm]")
    axes[0].set_title("(a) Bias -- not removed by filtering")

    axes[1].set_xlabel("Distance [cm]")
    axes[1].set_ylabel("Standard deviation [mm]")
    axes[1].set_title("(b) Noise -- reduced by filtering")

    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fname}.pdf")


# ---------- Figure 3: error distribution ----------
def plot_error_distribution(results, fname="images/depth_error_distribution"):
    labels = np.array(results["labels_cm"]) / 100.0
    methods, data, colors = [], [], []

    for m, method_data in results["methods"].items():
        pred = np.array(method_data["pred"])
        err = (pred - labels) * 1000
        err = err[np.isfinite(err)]
        if err.size:
            methods.append(get_name(m))
            data.append(err)
            colors.append(COLORS.get(m, "gray"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Estimation error [mm]")
    ax.set_title("Distribution of errors relative to actual distance")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=12)

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fname}.pdf")


# ---------- LaTeX table ----------
def latex_table(results, fname="depth_comparison.tex"):
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\caption{Comparison of depth estimation methods at the hand "
                "wrist position. Errors are reported relative to the actual distance measured "
                "with a ruler.}\n")
        f.write("\\label{tab:depth-comparison}\n\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("\\textbf{Method} & \\textbf{Bias} & \\textbf{Noise (std)} & "
                "\\textbf{MAE} & \\textbf{Time} \\\\\n")
        f.write(" & [mm] & [mm] & [mm] & [ms/frame] \\\\\n\\hline\n")
        for m, data in results["methods"].items():
            o = data["vs_ruler"]["overall"]
            t = results.get("timings_ms_per_frame", {}).get(m, float("nan"))
            t_str = f"{t:.1f}" if np.isfinite(t) else "--"
            f.write(f"{get_name(m)} & {o['bias_mm']:.1f} & {o['std_mm']:.1f} & "
                    f"{o['mae_mm']:.1f} & {t_str} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved: {fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="rezultati_dubina.json")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    plot_estimated_vs_actual(results)
    plot_by_distance(results)
    plot_error_distribution(results)
    latex_table(results)