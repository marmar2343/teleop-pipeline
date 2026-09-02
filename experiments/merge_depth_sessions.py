"""
merge_depth_sessions.py

Merges results from multiple recorded sessions and shows variability
BETWEEN sessions, rather than only within a single session.

Why: a single session gives one value per method, but does not show
whether that value is repeatable. Multiple independent sessions allow
the mean and standard deviation between sessions to be reported.

Usage:
    # 1) Record multiple sessions
    python record_depth_session.py --out session_1.npz
    python record_depth_session.py --out session_2.npz
    python record_depth_session.py --out session_3.npz

    # 2) Process each session
    python compare_depth_methods.py --session session_1.npz --depth-scale 0.0001 --out results_1.json
    python compare_depth_methods.py --session session_2.npz --depth-scale 0.0001 --out results_2.json
    python compare_depth_methods.py --session session_3.npz --depth-scale 0.0001 --out results_3.json

    # 3) Merge sessions
    python merge_depth_sessions.py results_1.json results_2.json results_3.json
"""

import argparse
import json
import os

import numpy as np


def load_all(paths):
    sessions = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        r["_name"] = os.path.basename(p)
        sessions.append(r)
    return sessions


def compute_session_metrics(sessions):
    """Compute key metrics for each session and method."""
    out = {}
    for s in sessions:
        labels = np.array(s["labels_cm"])
        distances = np.array(sorted(set(labels.tolist())))

        for method, data in s["methods"].items():
            pred = np.array(data["pred"])
            overall = data["vs_ruler"]["overall"]

            # Mean prediction at each distance - used for sensitivity and R^2
            means = []
            for d in distances:
                sel = (labels == d) & np.isfinite(pred)
                means.append(np.mean(pred[sel]) * 100 if sel.sum() else np.nan)
            means = np.array(means)

            valid = np.isfinite(means)
            if valid.sum() >= 2:
                r2 = np.corrcoef(distances[valid], means[valid])[0, 1] ** 2
                # Sensitivity: slope of the fitted line
                slope = np.polyfit(distances[valid], means[valid], 1)[0]
                # Minimum local sensitivity -- reveals saturation
                min_local = np.min(np.diff(means[valid]) / np.diff(distances[valid]))
            else:
                r2 = slope = min_local = np.nan

            out.setdefault(method, []).append({
                "session": s["_name"],
                "mae_mm": overall["mae_mm"],
                "bias_mm": overall["bias_mm"],
                "std_mm": overall["std_mm"],
                "r2": r2,
                "slope": slope,
                "min_local_sensitivity": min_local,
                "ms": s.get("timings_ms_per_frame", {}).get(method, np.nan),
            })

    return out


def mean_std(vals, dec=1):
    v = np.array([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return "--"
    if v.size == 1:
        return f"{v[0]:.{dec}f}"
    return f"{v.mean():.{dec}f}±{v.std():.{dec}f}"


def print_results(by_method, n_sessions):
    print("\n" + "=" * 92)
    print(f"MERGED ACROSS {n_sessions} SESSIONS (mean ± std BETWEEN sessions)")
    print("=" * 92)
    print(f"{'Method':22s} {'MAE [mm]':>14s} {'Bias':>14s} {'Noise [mm]':>13s} "
          f"{'R^2':>13s} {'ms/frame':>10s}")
    print("-" * 92)

    for method, rows in by_method.items():
        print(f"{method:22s} "
              f"{mean_std([r['mae_mm'] for r in rows]):>14s} "
              f"{mean_std([r['bias_mm'] for r in rows]):>14s} "
              f"{mean_std([r['std_mm'] for r in rows]):>13s} "
              f"{mean_std([r['r2'] for r in rows], 3):>13s} "
              f"{mean_std([r['ms'] for r in rows]):>10s}")

    print("=" * 92)

    # -- Repeatability: R^2 range and minimum sensitivity --
    print("\n" + "=" * 92)
    print("INTER-SESSION REPEATABILITY")
    print("=" * 92)
    print(f"{'Method':22s} {'R^2 range':>20s} {'Minimum local sensitivity':>28s}")
    print(f"{'':22s} {'(min -- max)':>20s} {'(1.0 = ideal, <0 = reversed)':>28s}")
    print("-" * 92)

    for method, rows in by_method.items():
        r2s = np.array([r["r2"] for r in rows if np.isfinite(r["r2"])])
        mins = np.array([r["min_local_sensitivity"] for r in rows
                         if np.isfinite(r["min_local_sensitivity"])])
        r2_text = f"{r2s.min():.3f} -- {r2s.max():.3f}" if r2s.size else "--"
        min_text = f"{mins.min():+.2f} -- {mins.max():+.2f}" if mins.size else "--"
        print(f"{method:22s} {r2_text:>20s} {min_text:>28s}")

    print("=" * 92)
    print("R^2               -- how well the predictions follow the true distance (1.0 = perfect)")
    print("Local sensitivity -- change in prediction per unit change in true distance;")
    print("                     values close to 0 mean the method can no longer distinguish distances,")
    print("                     negative values mean the prediction changes in the opposite direction")

    # -- Individual sessions, for inspecting deviations --
    print("\n" + "=" * 92)
    print("INDIVIDUAL SESSIONS (MAE [mm] / R^2)")
    print("=" * 92)

    names = [r["session"] for r in list(by_method.values())[0]]
    print(f"{'Method':22s}" + "".join(f"{name[:16]:>18s}" for name in names))
    print("-" * 92)

    for method, rows in by_method.items():
        line = f"{method:22s}"
        for r in rows:
            line += f"{r['mae_mm']:>10.1f}/{r['r2']:>7.3f}"
        print(line)

    print("=" * 92)


def latex_table(by_method, n_sessions, fname="depth_comparison_table.tex"):
    NAMES = {
        "realsense": "RealSense (reference)",
        "yolo": "YOLO26-depth",
        "depthany": "Depth Anything V2",
        "mp_size": "Apparent hand size",
        "yolo_poravnat": "YOLO26-depth, aligned",
        "depthany_poravnat": "Depth Anything V2, aligned",
        "mp_size_poravnat": "Apparent hand size, aligned",
    }

    def latex_mean_std(vals, dec=1):
        v = np.array([x for x in vals if np.isfinite(x)])
        if v.size == 0:
            return "--"
        if v.size == 1:
            return f"${v[0]:.{dec}f}$".replace(".", "{,}")
        return f"${v.mean():.{dec}f} \\pm {v.std():.{dec}f}$".replace(".", "{,}")

    with open(fname, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\caption{Depth estimation accuracy with respect to the true "
                "distance measured using a ruler. Mean values and standard "
                f"deviations across {n_sessions} independent recording sessions are reported.}}\n")
        f.write("\\label{tab:depth-comparison}\n\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("\\textbf{Method} & \\textbf{Bias} & \\textbf{Noise} & "
                "\\textbf{MAE} & \\textbf{Time} \\\\\n")
        f.write(" & [mm] & [mm] & [mm] & [ms/frame] \\\\\n\\hline\n")

        for method, rows in by_method.items():
            f.write(f"{NAMES.get(method, method)} & "
                    f"{latex_mean_std([r['bias_mm'] for r in rows])} & "
                    f"{latex_mean_std([r['std_mm'] for r in rows])} & "
                    f"{latex_mean_std([r['mae_mm'] for r in rows])} & "
                    f"{latex_mean_std([r['ms'] for r in rows])} \\\\\n")

        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"\nSaved: {fname}")


def plot_results(by_method, sessions, fname="images/depth_repeatability"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("images", exist_ok=True)

    COLORS = {"realsense": "#2e7d32", "yolo": "#1f77b4",
              "depthany": "#9467bd", "mp_size": "#d62728"}

    base_methods = [m for m in by_method if not m.endswith("_poravnat")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) prediction versus distance, all sessions
    ax = axes[0]

    for session in sessions:
        labels = np.array(session["labels_cm"])
        distances = np.array(sorted(set(labels.tolist())))

        for method in base_methods:
            if method not in session["methods"]:
                continue

            pred = np.array(session["methods"][method]["pred"])
            means = [np.mean(pred[(labels == d) & np.isfinite(pred)]) * 100
                     for d in distances]

            ax.plot(distances, means, "o-", color=COLORS.get(method, "gray"),
                    alpha=0.55, markersize=4)

    lo, hi = distances.min() - 5, distances.max() + 5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("True distance [cm]")
    ax.set_ylabel("Estimated distance [cm]")
    ax.set_title("(a) All sessions -- dashed line is ideal")
    ax.grid(alpha=0.3)
    ax.legend([plt.Line2D([], [], color=COLORS.get(m, "gray")) for m in base_methods],
              base_methods, fontsize=8)

    # (b) MAE per session - inter-session variability
    ax = axes[1]
    x = np.arange(len(by_method))

    for i, (method, rows) in enumerate(by_method.items()):
        vals = [r["mae_mm"] for r in rows]
        ax.scatter([i] * len(vals), vals, s=45,
                   color=COLORS.get(method.replace("_poravnat", ""), "gray"),
                   alpha=0.75, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.mean(vals)] * 2,
                "k-", linewidth=2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(list(by_method), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("MAE [mm]")
    ax.set_title("(b) Inter-session variability (line = mean)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fname}.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="results_*.json files")
    ap.add_argument("--table", default="depth_comparison_table.tex")
    args = ap.parse_args()

    sessions = load_all(args.results)
    by_method = compute_session_metrics(sessions)

    print_results(by_method, len(sessions))
    latex_table(by_method, len(sessions), args.table)
    plot_results(by_method, sessions)