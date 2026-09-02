"""
compare_controllers.py

Comparison of the custom DLS_IK_Solver with RoboSuite's built-in OSC_POSE
controller, using an IDENTICAL, scripted sequence of waypoints derived
from the TRUE optimal solution of the Towers of Hanoi (7 moves for 3 disks).

INTENTIONALLY without live teleoperation -- using identical input for both
controllers is the only way to fairly attribute measured differences to
the controller itself rather than to operator variability.
"""
import numpy as np

# 1. Generate optimal move sequence (classic Hanoi algorithm)

def hanoi_moves(n, source, target, aux):
    """
    Returns a list of moves (disk_size, from_peg, to_peg) for moving n disks from the source peg to the target peg using the auxiliary peg.
    disk_size: 1 = smallest (cubeC), 2 = medium (cubeB), 3 = largest (cubeA)
    """
    if n == 0:
        return []
    moves = []
    moves += hanoi_moves(n - 1, source, aux, target)
    moves.append((n, source, target))
    moves += hanoi_moves(n - 1, aux, target, source)
    return moves


DISK_TO_CUBE = {1: "cubeC", 2: "cubeB", 3: "cubeA"}

# OSC_POSE input scaling - must match output_max from the BASIC configuration
OSC_POS_SCALE = 0.05   # m
OSC_ROT_SCALE = 0.5    # rad


# 2. Generate waypoints from the move sequence

def generate_waypoints(env, source_peg_idx, target_peg_idx, approach_height=0.08, lift_height=0.05):
    """
    Generates the FULL sequence of waypoints required to solve the task.

    The current state of all towers is tracked so that the correct z-heights
    for picking up and placing each disk can be calculated at every move.

    Returns a list of dictionaries, one for each waypoint:
        {"pos": np.array(3,), "phase": str, "move_idx": int, "gripper": float}
    """
    aux_peg_idx = [0, 1, 2]
    aux_peg_idx.remove(source_peg_idx)
    aux_peg_idx.remove(target_peg_idx)
    aux_peg_idx = aux_peg_idx[0]

    moves = hanoi_moves(3, source_peg_idx, target_peg_idx, aux_peg_idx)

    # Track the current state of each tower, ordered from bottom to top
    peg_stacks = {source_peg_idx: ["cubeA", "cubeB", "cubeC"], target_peg_idx: [], aux_peg_idx: []}

    h = env.half_heights
    z0 = env.table_offset[2]

    waypoints = []

    for move_idx, (disk_size, from_peg, to_peg) in enumerate(moves):
        cube_name = DISK_TO_CUBE[disk_size]

        # Remove the disk from the top of the source peg
        assert peg_stacks[from_peg][-1] == cube_name, f"Potez {move_idx}: {cube_name} nije na vrhu klina {from_peg}"
        peg_stacks[from_peg].pop()

        from_xy = env._peg_world_xy(from_peg)
        to_xy = env._peg_world_xy(to_peg)

        # Pickup height: top of the remaining stack on the source peg
        pickup_z = z0 + sum(2 * h[c] for c in peg_stacks[from_peg]) + h[cube_name]

        # Place height: top of the stack on the target peg before placement
        place_z = z0 + sum(2 * h[c] for c in peg_stacks[to_peg]) + h[cube_name]

        # Safe transfer height: above the tallest stack currently in the scene.
        # Without this, diagonal transfer paths may collide with existing disks.
        tallest_stack_top = z0
        for stack in peg_stacks.values():
            top = z0 + sum(2 * h[c] for c in stack)
            tallest_stack_top = max(tallest_stack_top, top)
        safe_z = max(pickup_z, place_z, tallest_stack_top) + approach_height

        seq = [
            {"pos": np.array([from_xy[0], from_xy[1], pickup_z + approach_height]), "phase": "pre-grasp", "gripper": -1.0},
            {"pos": np.array([from_xy[0], from_xy[1], pickup_z]), "phase": "grasp", "gripper": -1.0},
            {"pos": np.array([from_xy[0], from_xy[1], pickup_z]), "phase": "grip", "gripper": 1.0},
            {"pos": np.array([from_xy[0], from_xy[1], safe_z]), "phase": "lift", "gripper": 1.0},
            {"pos": np.array([to_xy[0], to_xy[1], safe_z]), "phase": "transfer", "gripper": 1.0},
            {"pos": np.array([to_xy[0], to_xy[1], place_z]), "phase": "lower", "gripper": 1.0},
            {"pos": np.array([to_xy[0], to_xy[1], place_z]), "phase": "release", "gripper": -1.0},
            {"pos": np.array([to_xy[0], to_xy[1], safe_z]), "phase": "retract", "gripper": -1.0},
        ]
        for wp in seq:
            wp["move_idx"] = move_idx
            wp["cube"] = cube_name
        waypoints.extend(seq)

        peg_stacks[to_peg].append(cube_name)

    return waypoints, moves


# 3. Run controller through waypoints and collect measurements


def _build_env(controller_backend, source_peg_idx, target_peg_idx, control_freq=30):
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))

    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    import hanoi_three_env  

    cc = load_composite_controller_config(controller="BASIC")
    if controller_backend == "DLS":
        cc["body_parts"]["right"]["type"] = "JOINT_POSITION"
        cc["body_parts"]["right"]["input_type"] = "absolute"
        cc["body_parts"]["right"]["interpolation"] = "linear"
    elif controller_backend == "OSC_POSE":
        pass  # BASIC already provides OSC_POSE in delta mode
    else:
        raise ValueError(f"Nepoznat backend: {controller_backend}")

    return suite.make(
        env_name="HanoiThree", robots="Panda", controller_configs=cc,
        source_peg_idx=source_peg_idx, target_peg_idx=target_peg_idx,
        randomize_pegs=False, color_code_pegs=False, hard_reset=False,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=control_freq, horizon=100000, ignore_done=True,
    )


def run_trial(controller_backend, source_peg_idx=0, target_peg_idx=2, max_steps_per_wp=400, pos_tol_mm=8.0, control_freq=30, verbose=False):
    """
    Runs one controller through the FULL waypoint sequence for solving Hanoi.

    max_steps_per_wp: maximum number of control steps allowed per waypoint.
    If the target is not reached within this limit, the test continues and
    records the waypoint as a failed convergence.

    pos_tol_mm: positional convergence tolerance in millimeters.
    """
    import time
    import robosuite.utils.transform_utils as T
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
    from ik_solver import DLS_IK_Solver

    env = _build_env(controller_backend, source_peg_idx, target_peg_idx, control_freq)
    env.reset()

    solver = DLS_IK_Solver(env, arm="right", damping=0.05, step_size=0.5, max_joint_step=0.2)
    locked_orn = solver.get_eef_orientation()

    robot = env.robots[0]
    qvel_idx = np.array([env.sim.model.get_joint_qvel_addr(n) for n in robot.robot_model.joints])
    joint_limits = np.array([env.sim.model.jnt_range[env.sim.model.joint_name2id(n)] for n in robot.robot_model.joints])

    waypoints, moves = generate_waypoints(env, source_peg_idx, target_peg_idx)

    m = {
        "step_err_mm": [], "step_time_ms": [], "step_qvel": [],
        "step_orn_err_deg": [], "step_min_limit_margin": [],
        "wp_converge_steps": [], "wp_final_err_mm": [], "wp_converged": [],
        "wp_phase": [], "wp_move_idx": [],
    }

    total_steps = 0

    for wp in waypoints:
        target = wp["pos"]
        gripper = np.array([wp["gripper"]])
        converged = False
        steps_used = max_steps_per_wp
        last_err = None

        # Gripper phases do not depend on positional convergence because the
        # robot is already at the target. A fixed duration allows the gripper
        # enough time to mechanically close or open.
        is_gripper_phase = wp["phase"] in ("grip", "release")
        n_steps = 25 if is_gripper_phase else max_steps_per_wp

        for step_i in range(n_steps):
            t0 = time.perf_counter()

            if controller_backend == "DLS":
                q_target = solver.step(solver.get_current_qpos(), target)
                action = np.concatenate([q_target, gripper])
            else:
                # OSC_POSE expects normalized actions in [-1, 1]. These are
                # internally scaled to output_max (0.05 m position, 0.5 rad
                # rotation). Raw metric values would therefore be scaled down.
                delta_pos = (target - solver.get_eef_position()) / OSC_POS_SCALE
                delta_rot = T.get_orientation_error(locked_orn, solver.get_eef_orientation()) / OSC_ROT_SCALE
                action = np.concatenate([
                    np.clip(np.concatenate([delta_pos, delta_rot]), -1.0, 1.0),
                    gripper,
                ])

            env.step(action)
            step_ms = (time.perf_counter() - t0) * 1000
            total_steps += 1

            err_mm = np.linalg.norm(target - solver.get_eef_position()) * 1000
            orn_err_deg = np.degrees(np.linalg.norm(
                T.get_orientation_error(locked_orn, solver.get_eef_orientation())))
            q_now = solver.get_current_qpos()
            margin = np.min(np.minimum(q_now - joint_limits[:, 0], joint_limits[:, 1] - q_now))

            m["step_err_mm"].append(err_mm)
            m["step_time_ms"].append(step_ms)
            m["step_qvel"].append(np.linalg.norm(env.sim.data.qvel[qvel_idx]))
            m["step_orn_err_deg"].append(orn_err_deg)
            m["step_min_limit_margin"].append(margin)
            last_err = err_mm

            if is_gripper_phase:
                converged = True           
                steps_used = step_i + 1
            elif err_mm < pos_tol_mm:
                converged = True
                steps_used = step_i + 1
                break

        m["wp_converge_steps"].append(steps_used)
        m["wp_final_err_mm"].append(last_err)
        m["wp_converged"].append(converged)
        m["wp_phase"].append(wp["phase"])
        m["wp_move_idx"].append(wp["move_idx"])

        if verbose:
            status = "OK" if converged else "FAILED"
            print(f"  move {wp['move_idx']+1} {wp['phase']:12s}: {steps_used:3d} steps, "
                  f"error {last_err:6.2f}mm [{status}]")

    m["success"] = bool(env._check_success())
    legal, reason = env.check_hanoi_legality()
    m["legal"] = bool(legal)
    m["illegal_reason"] = reason
    m["total_steps"] = total_steps
    m["backend"] = controller_backend
    m["source_peg"] = source_peg_idx
    m["target_peg"] = target_peg_idx

    env.close()
    return m


if __name__ == "__main__":
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"

    for backend in ["DLS", "OSC_POSE"]:
        print(f"\n{'='*60}\n{backend}\n{'='*60}")
        r = run_trial(backend, 0, 2, verbose=False)
        conv_rate = 100 * np.mean(r["wp_converged"])
        print(f"Task success:          {r['success']}")
        print(f"Legal final state:     {r['legal']}")
        print(f"Waypoint convergence:  {conv_rate:.0f}% ({sum(r['wp_converged'])}/{len(r['wp_converged'])})")
        print(f"Average steps/point:   {np.mean(r['wp_converge_steps']):.1f}")
        print(f"Average position error:{np.mean(r['step_err_mm']):.2f} mm")
        print(f"Average orientation:   {np.mean(r['step_orn_err_deg']):.3f} deg")
        print(f"Time per step:         {np.mean(r['step_time_ms']):.3f} ms")
        print(f"Total steps:           {r['total_steps']}")


# 4. Full comparison: all 6 source/target combinations

def run_full_comparison(save_path="controller_comparison_results.json", n_repeats=5):
    """
    n_repeats: number of repetitions for EACH peg combination.

    Repetitions are useful because HanoiThree applies a random displacement
    to the disks after each reset (up to +-5 mm), so trials are not identical.
    """
    import json
    import itertools

    combos = [(s, t) for s, t in itertools.permutations([0, 1, 2], 2)]
    results = []
    total = len(combos) * n_repeats * 2

    for backend in ["DLS", "OSC_POSE"]:
        for src_peg, tgt_peg in combos:
            for rep in range(n_repeats):
                print(f"  [{len(results)+1:3d}/{total}] {backend:9s} peg {src_peg}->{tgt_peg} "
                      f"rep.{rep+1} ...", end=" ", flush=True)
                r = run_trial(backend, src_peg, tgt_peg)
                r["repeat"] = rep
                print(f"success={r['success']}  convergence={100*np.mean(r['wp_converged']):.0f}%  "
                      f"steps={r['total_steps']}")
                results.append(r)

            # Save after every peg combination so that completed measurements are not lost if the long comparison is interrupted
            with open(save_path, "w") as f:
                json.dump(results, f)

    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"\nSaved to {save_path}")
    return results