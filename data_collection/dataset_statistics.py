"""
dataset_statistics.py

Computes statistics for the collected demonstration dataset (demo.hdf5)
and plots the end-effector trajectories through space.

What it computes:
  - number of successful demonstrations
  - episode length in steps and seconds (mean, std, min, max)
  - success rate
  - fraction of time the gripper is closed
  - end-effector path length per episode

NOTE on success rate:
demo.hdf5 contains ONLY successful episodes -- unsuccessful ones are not
stored. The success rate can still be computed, because the
"processed_episodes" attribute records ALL processed episodes (both
successful and unsuccessful), so the ratio of stored demonstrations to
the length of that list gives the desired value. If that attribute is
missing (older files), the rate is not reported -- better than guessing.

Usage:
    python dataset_statistics.py demonstration/hanoi_dataset/demo.hdf5
"""

import argparse
import json
import os

import h5py
import numpy as np

CONTROL_FREQ = 30.0


def load_dataset(path):
    with h5py.File(path, "r") as f:
        data = f["data"]
        demo_names = sorted(data.keys(), key=lambda n: int(n.split("_")[1]))
 
        info = {
            "env": data.attrs.get("env", "?"),
            "date": f"{data.attrs.get('date', '?')} {data.attrs.get('time', '')}".strip(),
            "n_demos": len(demo_names),
        }
 
        # Success rate - see note in the module docstring
        if "processed_episodes" in data.attrs:
            processed = json.loads(data.attrs["processed_episodes"])
            info["n_processed"] = len(processed)
            info["success_rate"] = (100.0 * len(demo_names) / len(processed) if processed else np.nan)
        else:
            info["n_processed"] = None
            info["success_rate"] = None
 
        episodes = []
 
        for name in demo_names:
            g = data[name]
 
            episodes.append({
                "name": name,
                "states": g["states"][()],
                "actions": g["actions"][()],
                "model_xml": g.attrs["model_file"],
            })
 
 
        env_info = json.loads(data.attrs["env_info"]) if "env_info" in data.attrs else None
 
    return info, episodes, env_info

def reconstruct_eef_trajectories(episodes, env_info, max_episodes=None):
    """
    Reconstructs the end-effector trajectory from the saved states.

    The states contain joint angles, not the end-effector position, so each
    state is set in the simulation and the actual end-effector position is
    obtained through forward kinematics.
    """

    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"

    import robosuite as suite

    # pipeline/ contains hanoi_three_env.py - add it to the import path regardless of which directory this script is run from
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))

    import hanoi_three_env  # noqa: F401

    env = suite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=int(CONTROL_FREQ),
        hard_reset=False,
    )

    env.reset()

    eef_site_id = env.robots[0].eef_site_id["right"]
    trajectories = []
    selected = (episodes[:max_episodes] if max_episodes else episodes)

    for ep in selected:
        xml = env.edit_model_xml(ep["model_xml"])

        env.reset_from_xml_string(xml)
        env.sim.reset()

        points = []

        for state in ep["states"]:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            points.append(np.copy(env.sim.data.site_xpos[eef_site_id]))

        trajectories.append(np.array(points))
    env.close()

    return trajectories


def compute_statistics(info, episodes, trajectories=None):
    lengths = np.array([len(ep["states"]) for ep in episodes])

    durations = lengths / CONTROL_FREQ

    # Fraction of steps during which the gripper is closed (last action component)
    closed_pct = []

    for ep in episodes:
        g = ep["actions"][:, -1]
        closed_pct.append(100.0 * np.mean(g > 0))

    closed_pct = np.array(closed_pct)

    s = {
        "n_demos": info["n_demos"],
        "n_processed": info["n_processed"],
        "success_rate": info["success_rate"],

        "steps_mean": lengths.mean(),
        "steps_std": lengths.std(),
        "steps_min": lengths.min(),
        "steps_max": lengths.max(),

        "seconds_mean": durations.mean(),
        "seconds_std": durations.std(),
        "seconds_min": durations.min(),
        "seconds_max": durations.max(),

        "total_steps": int(lengths.sum()),
        "total_minutes": durations.sum() / 60.0,

        "grip_closed_mean": closed_pct.mean(),
        "grip_closed_std": closed_pct.std(),
    }

    if trajectories:
        path_lengths = [np.sum(np.linalg.norm(np.diff(p, axis=0),axis=1)) for p in trajectories]

        s["path_mean"] = float(np.mean(path_lengths))

        s["path_std"] = float(np.std(path_lengths))

    return s


def print_summary(s, info):
    print("=" * 62)
    print(f"DATASET STATISTICS   ({info['env']}, {info['date']})")
    print("=" * 62)
    print(f"Successful demonstrations        : {s['n_demos']}")

    if s["success_rate"] is not None:
        print(f"Total attempts (processed)       : {s['n_processed']}")
        print(f"Success rate                     : {s['success_rate']:.1f} %")
    else:
        print("Success rate                     : unavailable (file has no processed_episodes attribute)")

    print()
    print(f"Episode length [steps]           : {s['steps_mean']:.0f} +- {s['steps_std']:.0f} (min {s['steps_min']}, max {s['steps_max']})")
    print(f"Episode length [s]                : {s['seconds_mean']:.1f} +- {s['seconds_std']:.1f} (min {s['seconds_min']:.1f}, max {s['seconds_max']:.1f})")
    print(f"Total steps in dataset            : {s['total_steps']}")
    print(f"Total duration                    : {s['total_minutes']:.1f} min")

    print()
    print(f"Fraction of steps gripper closed  : {s['grip_closed_mean']:.1f} +- {s['grip_closed_std']:.1f} %")

    if "path_mean" in s:
        print(f"End-effector path length/episode  : {s['path_mean']:.2f} +- {s['path_std']:.2f} m")

    print("=" * 62)


def plot_trajectories(trajectories, peg_y=None, table_z=None, fname="plots/end_effector_trajectories"):
    """
    Plot end-effector trajectories.

    VIEW CHOICE: The pegs are arranged along the y axis, while
    picking up and setting down happens along z. A top-down view (x-y)
    therefore squashes everything into a line, since x is nearly
    constant. The side view (y-z) shows the "lift-transfer-lower"
    arcs, which are the essence of the task, so that view is used.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    os.makedirs("plots", exist_ok=True)
    fig = plt.figure(figsize=(14, 4.8))

    # ---- (a) One episode, color = elapsed time ----
    ax = fig.add_subplot(131)
    p = trajectories[0]
    t = np.linspace(0, 1, len(p))

    for i in range(len(p) - 1):
        ax.plot(p[i:i+2, 1], p[i:i+2, 2], color=plt.cm.viridis(t[i]), linewidth=1.4)

    if peg_y is not None:
        for y in peg_y:
            ax.axvline(y, color="#bbbbbb", linestyle=":", linewidth=1, zorder=0)

    if table_z is not None:
        ax.axhline(table_z, color="#666666", linewidth=1.2, zorder=0)

    ax.set_xlabel("y [m] (direction between pegs)")
    ax.set_ylabel("z [m] (height)")
    ax.set_title("(a) One episode, color = elapsed time")
    ax.grid(alpha=0.25)

    # ---- (b) A few episodes, side view ----
    ax2 = fig.add_subplot(132)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(trajectories), 5)))

    for p, c in zip(trajectories[:5], colors):
        ax2.plot(p[:, 1], p[:, 2], color=c, linewidth=0.9, alpha=0.75)

    if peg_y is not None:
        for y in peg_y:
            ax2.axvline(y, color="#bbbbbb", linestyle=":", linewidth=1, zorder=0)

    if table_z is not None:
        ax2.axhline(table_z, color="#666666", linewidth=1.2, zorder=0)

    ax2.set_xlabel("y [m]")
    ax2.set_ylabel("z [m]")
    ax2.set_title(f"(b) {min(len(trajectories), 5)} episodes, side view")
    ax2.grid(alpha=0.25)

    # ---- (c) Height distribution ----
    ax3 = fig.add_subplot(133)
    all_z = np.concatenate([p[:, 2] for p in trajectories])
    ax3.hist(all_z, bins=50, color="#4c72b0", alpha=0.85, orientation="horizontal")

    if table_z is not None:
        ax3.axhline(table_z, color="#666666", linewidth=1.2)

    ax3.set_ylabel("z [m]")
    ax3.set_xlabel("Number of steps")
    ax3.set_title("(c) End-effector height distribution")
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fname}.pdf")


def write_latex_table(s, fname="dataset_statistics.tex"):
    """
    Write the dataset statistics to a LaTeX table.

    The table is written in English so that it can be directly
    included in the English version of the thesis.
    """
    def fmt(x, dec=1):
        return f"${x:.{dec}f}$".replace(".", "{,}")

    with open(fname, "w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[H]\n"
            "\\caption{Basic statistics of the collected demonstration dataset.}\n"
            "\\label{tab:dataset-stat}\n"
            "\\centering\n"
        )
        f.write("\\begin{tabular}{lc}\n\\hline\n")
        f.write("\\textbf{Measure} & \\textbf{Value} \\\\\n\\hline\n")
        f.write(f"Number of successful demonstrations & ${s['n_demos']}$ \\\\\n")

        if s["success_rate"] is not None:
            f.write(f"Total number of attempts & ${s['n_processed']}$ \\\\\n")
            f.write(f"Success rate & {fmt(s['success_rate'])}~\\% \\\\\n")

        f.write(
            f"Mean episode length [steps] & "
            f"${s['steps_mean']:.0f} \\pm {s['steps_std']:.0f}$ \\\\\n"
        )
        f.write(
            f"Mean episode length [s] & "
            f"${s['seconds_mean']:.1f} \\pm {s['seconds_std']:.1f}$ \\\\\n"
            .replace(".", "{,}")
        )
        f.write(f"Total number of steps & ${s['total_steps']}$ \\\\\n")
        f.write(f"Total duration [min] & {fmt(s['total_minutes'])} \\\\\n")
        f.write("Control frequency [Hz] & $30$ \\\\\n")
        f.write(
            f"Fraction of steps with closed gripper [\\%] & "
            f"${s['grip_closed_mean']:.1f} \\pm {s['grip_closed_std']:.1f}$ \\\\\n"
            .replace(".", "{,}")
        )

        if "path_mean" in s:
            f.write(
                f"End-effector path length per episode [m] & "
                f"${s['path_mean']:.2f} \\pm {s['path_std']:.2f}$ \\\\\n"
                .replace(".", "{,}")
            )

        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"Saved: {fname}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",nargs="?",default="demonstrations/hanoi_dataset/demo.hdf5")
    parser.add_argument("--max-trajectories",type=int,default=15,help="how many episodes to plot (too many clutter the figure)")
    parser.add_argument("--no-trajectories",action="store_true",help="skip trajectory reconstruction (faster, robosuite not required)")

    args = parser.parse_args()
    info, episodes, env_info = load_dataset(args.dataset)
    trajectories = None

    if (not args.no_trajectories and env_info is not None):
        print(f"Reconstructing trajectories for {min(args.max_trajectories, len(episodes))} episodes...")
        trajectories = reconstruct_eef_trajectories(episodes,env_info,args.max_trajectories)

    s = compute_statistics(info,episodes,trajectories)
    print_summary(s,info)
    write_latex_table(s)

    if trajectories:
        # Peg positions and table height - used as reference lines in the plot
        peg_y = table_z = None

        try:
            import robosuite.macros as macros
            macros.IMAGE_CONVENTION = "opencv"

            import robosuite as suite
            import os
            import sys

            sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))

            import hanoi_three_env 

            e = suite.make(**env_info, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, hard_reset=False)

            e.reset()

            peg_y = [e._peg_world_xy(i)[1] for i in range(3)]
            table_z = e.table_offset[2]
            e.close()

        except Exception as ex:
            print(f"(note: reference lines skipped -- {ex})")

        plot_trajectories(trajectories,peg_y,table_z)