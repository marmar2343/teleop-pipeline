"""
plot_controller_comparison.py

Plots results from compare_controllers.run_full_comparison().
Input: controller_comparison_results.json
Output: images/comparison_*.pdf and .png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"DLS": "#1f77b4", "OSC_POSE": "#d62728"}
NAMES = {"DLS": "Custom (DLS)", "OSC_POSE": "\\texttt{OSC\\_POSE}"}
NAMES_PLAIN = {"DLS": "Custom (DLS)", "OSC_POSE": "RoboSuite OSC_POSE"}

os.makedirs("images", exist_ok=True)


def load_results(path="controller_comparison_results.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _group_by_backend(results):
    out = {}
    for r in results:
        out.setdefault(r["backend"], []).append(r)
    return out


# ---------- Figure 1: overall comparison (4 panels) ----------
def plot_overall(results, fname="images/comparison_overall"):
    groups = _group_by_backend(results)
    backends = list(groups.keys())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # (a) task success rate
    ax = axes[0]
    vals = [100 * np.mean([r["success"] for r in groups[b]]) for b in backends]
    ax.bar([NAMES_PLAIN[b] for b in backends], vals, color=[COLORS[b] for b in backends])
    ax.set_ylabel("Task success [%]")
    ax.set_title("(a) Task success")
    ax.set_ylim(0, 105)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)

    # (b) waypoint convergence rate
    ax = axes[1]
    vals = [100 * np.mean([np.mean(r["wp_converged"]) for r in groups[b]]) for b in backends]
    ax.bar([NAMES_PLAIN[b] for b in backends], vals, color=[COLORS[b] for b in backends])
    ax.set_ylabel("Reached waypoints [%]")
    ax.set_title("(b) Convergence")
    ax.set_ylim(0, 105)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)

    # (c) average number of steps per waypoint
    ax = axes[2]
    data = [[np.mean(r["wp_converge_steps"]) for r in groups[b]] for b in backends]
    bp = ax.boxplot(data, tick_labels=[NAMES_PLAIN[b] for b in backends], patch_artist=True)
    for patch, b in zip(bp["boxes"], backends):
        patch.set_facecolor(COLORS[b])
        patch.set_alpha(0.6)
    ax.set_ylabel("Steps per waypoint")
    ax.set_title("(c) Response speed")

    # (d) orientation error
    ax = axes[3]
    data = [[np.mean(r["step_orn_err_deg"]) for r in groups[b]] for b in backends]
    bp = ax.boxplot(data, tick_labels=[NAMES_PLAIN[b] for b in backends], patch_artist=True)
    for patch, b in zip(bp["boxes"], backends):
        patch.set_facecolor(COLORS[b])
        patch.set_alpha(0.6)
    ax.set_ylabel("Orientation error [°]")
    ax.set_title("(d) Orientation stability")


    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fname}.pdf")


# ---------- Figure 2: error over time (one representative trial) ----------
def plot_error_over_time(results, src=0, tgt=2, fname="images/comparison_error_time"):
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for r in results:
        if r["source_peg"] == src and r["target_peg"] == tgt:
            err = np.array(r["step_err_mm"])
            ax.plot(np.arange(len(err)), err, color=COLORS[r["backend"]],
                    label=NAMES_PLAIN[r["backend"]], linewidth=1.0, alpha=0.85)

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Distance to current waypoint [mm]")
    ax.set_title(f"Waypoint tracking error during task execution (peg {src} → {tgt})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fname}.pdf")


# ---------- Grafik 3: koraci po fazi poteza ----------
def plot_by_phase(results, fname="images/controller_comparison_phases"):
    groups = _group_by_backend(results)
    phases = ["pre-grasp", "grasp", "grip", "lift",
              "transfer", "lower", "release", "retract"]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(phases))
    w = 0.35

    for i, b in enumerate(groups):
        means = []
        for phase in phases:
            vals = []
            for r in groups[b]:
                vals += [s for s, p in zip(r["wp_converge_steps"], r["wp_phase"]) if p == phase]
            means.append(np.mean(vals) if vals else 0)
        ax.bar(x + (i - 0.5) * w, means, w, label=NAMES_PLAIN[b],
               color=COLORS[b], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=20)
    ax.set_ylabel("Average number of steps")
    ax.set_title("Number of steps per movement phase (average across all peg combinations)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fname}.pdf")


# ---------- LaTeX table ----------
def _decimal_comma(x, dec=1):
    """Formats a number with a decimal comma for Serbian notation."""
    return f"${x:.{dec}f}".replace(".", "{,}") + "$"


def _mean_std(vals, dec=1):
    """Formats mean ± standard deviation with a decimal comma."""
    m, s = float(np.mean(vals)), float(np.std(vals))
    if len(vals) < 2:
        return _decimal_comma(m, dec)
    return f"${m:.{dec}f} \\pm {s:.{dec}f}$".replace(".", "{,}")


def latex_table(results, fname="controller_comparison_table.tex"):
    groups = _group_by_backend(results)
    n_per_backend = {b: len(groups[b]) for b in groups}

    with open(fname, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\caption{Comparison of controllers using a scripted Hanoi Towers "
                "sequence. Mean values and standard deviations are reported across "
                "all trials (six peg combinations with repetitions).}\n")
        f.write("\\label{tab:controller-comparison}\n\\centering\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("\\textbf{Controller} & \\textbf{Success} & \\textbf{Conv.} & "
                "\\textbf{Steps/wp.} & \\textbf{Error} & \\textbf{Orient.} & "
                "\\textbf{Time/step} \\\\\n")
        f.write(" & [\\%] & [\\%] & & [mm] & [°] & [ms] \\\\\n\\hline\n")

        for b in groups:
            rs = groups[b]
            success = 100 * np.mean([r["success"] for r in rs])
            conv = 100 * np.mean([np.mean(r["wp_converged"]) for r in rs])
            steps = [np.mean(r["wp_converge_steps"]) for r in rs]
            error = [np.mean(r["step_err_mm"]) for r in rs]
            orientation = [np.mean(r["step_orn_err_deg"]) for r in rs]
            time = [np.mean(r["step_time_ms"]) for r in rs]

            f.write(f"{NAMES[b]} & {success:.0f} & {conv:.0f} & "
                    f"{_mean_std(steps, 1)} & {_mean_std(error, 1)} & "
                    f"{_mean_std(orientation, 2)} & {_mean_std(time, 1)} \\\\\n")

        f.write("\\hline\n\\end{tabular}\n")
        n_txt = ", ".join(f"{NAMES[b]}: $n={n_per_backend[b]}$" for b in groups)
        f.write(f"\\\\[2pt]\n\\footnotesize Number of trials -- {n_txt}.\n")
        f.write("\\end{table}\n")

    print(f"Saved: {fname}")


if __name__ == "__main__":
    results = load_results()
    plot_overall(results)
    plot_error_over_time(results)
    plot_by_phase(results)
    latex_table(results)

