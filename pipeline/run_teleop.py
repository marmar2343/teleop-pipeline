"""
run_teleop.py

Phase 4: Collection of human teleoperation demonstrations for the VLA dataset.

The script integrates the complete teleoperation pipeline:

    MediaPipeDevice
        -> 3D hand position from RealSense depth
        -> rate-control velocity command
        -> DLS_IK_Solver
        -> joint-position command
        -> HanoiThree environment
        -> DataCollectionWrapper
        -> HDF5 demonstration dataset

The robot is controlled through a JOINT_POSITION controller. The hand
displacement is converted into a Cartesian velocity command, which is
integrated over time to obtain the desired end-effector position. The DLS
inverse kinematics solver then converts this Cartesian target into joint
positions.

Demonstrations are recorded using RoboSuite's DataCollectionWrapper.
Only demonstrations that reach the environment's success condition are
stored in the final HDF5 dataset.

Hanoi rule validation is performed online for monitoring purposes, but it
does not affect whether a demonstration is recorded. The complete sequence
is validated separately by validate_dataset.py after data collection.
"""

import datetime
import json
import os
import time
from glob import glob

import cv2
import h5py
import numpy as np
import robosuite as suite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"  # Use the OpenCV image convention; this must be set before suite.make()

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from mediapipe_device import MediaPipeDevice
from ik_solver import DLS_IK_Solver
import hanoi_three_env

# Teleoperation mapping and velocity scaling

# These constants map the RealSense camera-frame displacement stored in
# device.stick to Cartesian velocity commands expressed in the robot frame.
#
# RealSense camera convention:
#   X: right
#   Y: down
#   Z: forward, away from the camera
SIGN_FORWARD = 1.0
SIGN_LATERAL = -1.0
SIGN_VERTICAL = -1.0

# Velocity gains for the lateral/vertical and depth directions
K_XY = 2.0
K_Z = 2.0

MAX_LINEAR_SPEED = 0.3

# Nonlinear response curve
# A nonlinear response curve provides finer control near the reference point
# while preserving a strong response for larger hand displacements.
#
# exponent = 1.0 -> linear response
# exponent = 2.0 -> quadratic response
# larger exponent -> stronger suppression of small displacements
CURVE_EXPONENT = 2.0

STICK_MAX_XY = 0.15
STICK_MAX_Z = 0.15


SUCCESS_HOLD_STEPS = 10


def _apply_curve(value, max_expected, exponent):
    normalized = np.clip(abs(value) / max_expected, 0.0, 1.0)
    shaped = normalized ** exponent
    return np.sign(value) * shaped * max_expected


def stick_to_velocity(stick):
    """
    Convert the 3D hand displacement into a Cartesian end-effector velocity.
    """
    shaped_x = _apply_curve(stick[0], STICK_MAX_XY, CURVE_EXPONENT)
    shaped_y = _apply_curve(stick[1], STICK_MAX_XY, CURVE_EXPONENT)
    shaped_z = _apply_curve(stick[2], STICK_MAX_Z, CURVE_EXPONENT)

    velocity = np.array([
        shaped_z * K_Z * SIGN_FORWARD,
        shaped_x * K_XY * SIGN_LATERAL,
        shaped_y * K_XY * SIGN_VERTICAL,
    ])

    speed = np.linalg.norm(velocity)
    if speed > MAX_LINEAR_SPEED:
        velocity = velocity * (MAX_LINEAR_SPEED / speed)

    return velocity


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Merge recorded episode files into a persistent HDF5 dataset.

    The temporary episode data produced by DataCollectionWrapper is stored as
    .npz files. This function scans those files, keeps only successful
    episodes, and appends them to demo.hdf5.

    The output structure is:

        data/
            date
            time
            repository_version
            env
            env_info
            demo_1/
                model_file
                states
                actions
            demo_2/
                model_file
                states
                actions
            ...

    Existing demonstrations are preserved. New successful demonstrations
    are appended using consecutive demo_N identifiers.

    A persistent list of processed episode directories is stored as an HDF5
    group attribute. This prevents the same temporary episode from being
    added more than once when the function is called repeatedly.
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")

    file_existed = os.path.exists(hdf5_path)
    f = h5py.File(hdf5_path, "a")  # "a" = append

    if "data" not in f:
        grp = f.create_group("data")
    else:
        grp = f["data"]

    # determine the next demonstration index from the existing dataset
    existing_nums = [int(k.split("_")[1]) for k in grp.keys() if k.startswith("demo_")]
    num_eps = max(existing_nums) if existing_nums else 0
    n_before = num_eps

    # Keep track of episode directories that have already been processed.
    # This makes repeated calls safe and prevents duplicate demonstrations.
    processed = set(json.loads(grp.attrs.get("processed_episodes", "[]")))

    env_name = grp.attrs.get("env", None)

    for ep_directory in sorted(os.listdir(directory)): 
        if ep_directory in processed:
            continue  

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        # The episode may not have been flushed to disk yet. Leave it
        # unprocessed so that it can be collected during the next call.
        if len(states) == 0:
            continue  

        processed.add(ep_directory)

        if success:
            print(f"[gather] {ep_directory}: SUCCESSFUL, saving to dataset")

            # DataCollectionWrapper stores the state after each action.
            # Therefore, the final state has no corresponding action and
            # must be removed before storing the trajectory.

            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # Store the MuJoCo model used to generate this demonstration so that the recorded state trajectory can be reproduced later.
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as xml_f:
                xml_str = xml_f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print(f"[gather] {ep_directory}: UNSUCCESSFUL, skipping")

    grp.attrs["processed_episodes"] = json.dumps(sorted(processed))

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()
    added = num_eps - n_before
    verb = "dopisano u postojeci" if file_existed else "sacuvano u nov"
    print(f"[gather] {verb} file -- {added} new demonstrations this time, "f"{num_eps} successful demonstrations total in {hdf5_path}")

    return hdf5_path


def build_env(control_freq=30, live_camera_names=("agentview",)):
    """
    Create and configure the HanoiThree environment for data collection.

    Args:
        control_freq (int): Control frequency in Hz.
        live_camera_names (tuple or None): Cameras rendered during recording
            for live monitoring. These images are not required by
            DataCollectionWrapper and are enabled only for visualization.

    Returns:
        tuple:
            env: Configured and wrapped RoboSuite environment.
            env_info: JSON-encoded environment configuration.

    Live camera rendering can significantly increase computational cost.
    Therefore, only the cameras required for monitoring should be enabled.
    Recorded demonstrations do not depend on these live camera streams.
    """
    controller_config = load_composite_controller_config(controller="BASIC")
    controller_config["body_parts"]["right"]["type"] = "JOINT_POSITION"
    controller_config["body_parts"]["right"]["input_type"] = "absolute"
    controller_config["body_parts"]["right"]["interpolation"] = "linear"

    config = {"env_name": "HanoiThree", "robots": "Panda", "controller_configs": controller_config, }

    live_camera_names = list(live_camera_names) if live_camera_names else None
    use_cameras = bool(live_camera_names)

    env = suite.make(
        **config,
        source_peg_idx=0,
        target_peg_idx=2,
        randomize_pegs=True, 
        color_code_pegs=True,
        has_renderer=True,
        hard_reset=False,  # Keep the same simulation object across environment resets.
                           # This is important because the IK solver maintains references to
                           # the current simulation model and data.
        has_offscreen_renderer=use_cameras,
        use_camera_obs=use_cameras,
        camera_names=live_camera_names,
        camera_heights=320 if use_cameras else None,
        camera_widths=320 if use_cameras else None,
        control_freq=control_freq,
        horizon=9000, # Allow sufficient time for a complete Tower of Hanoi demonstration,
                      # including object manipulation, transfers, and corrections.
        ignore_done=True,
    )

    env_info = json.dumps(config)

    env = VisualizationWrapper(env)

    return env, env_info


def collect_one_demonstration(env, device, ik_kwargs, dt):
    """
    Collect one teleoperation demonstration.

    The episode continues until the Hanoi task satisfies the environment's
    success condition for SUCCESS_HOLD_STEPS consecutive control steps, or
    until the environment horizon is reached.

    A fresh DLS_IK_Solver is created after every environment reset. This is
    necessary because DataCollectionWrapper may recreate the underlying
    simulation during reset, and the solver must reference the current
    simulation model and data.

    Hanoi legality is checked at every control step for monitoring. A single
    legality violation does not terminate or invalidate the episode during
    collection. The complete legality history is evaluated separately during
    dataset validation.

    Args:
        env: DataCollectionWrapper-wrapped HanoiThree environment.
        device: Active MediaPipe teleoperation device.
        ik_kwargs (dict): Keyword arguments for DLS_IK_Solver.
        dt (float): Control timestep in seconds.

    Returns:
        bool: True if the Hanoi legality condition was satisfied throughout
            the entire episode, otherwise False.
    """
    obs = env.reset()
    env.render()

    hanoi_env = env.env.env  # DataCollectionWrapper -> VisualizationWrapper -> HanoiThree

    solver = DLS_IK_Solver(env, **ik_kwargs)

    device.start_control()

    target_ee_pos = solver.get_eef_position()
    prev_clutch = False
    step_count = 0
    success_hold_count = 0
    episode_always_legal = True

    print(f"\n=== New demonstration -- source={hanoi_env.source_peg_idx}, " f"target={hanoi_env.target_peg_idx} ===")
    print("Hold SPACE to control the robot with your hand. " "Complete the task to finish the demonstration automatically.")

    while True:
        # Read the latest hand-control state atomically so that the control loop does not access partially updated data from the device thread.
        with device.lock:
            clutch_now = device._clutch_active
            stick = device.stick.copy()

        q_current = solver.get_current_qpos()

        if clutch_now and not prev_clutch:
            target_ee_pos = solver.get_eef_position()

        if clutch_now:
            velocity = stick_to_velocity(stick)

            # rate control: integrate the commanded Cartesian velocity over one control timestep to obtain the next end-effector target
            target_ee_pos = target_ee_pos + velocity * dt
            q_target = solver.step(q_current, target_ee_pos)
        else:
            q_target = q_current

        prev_clutch = clutch_now

        gripper_action = np.array([1.0 if device.grasp else -1.0])
        action = np.concatenate([q_target, gripper_action])

        assert len(action) == env.action_dim, (
            f"Ocekivano {env.action_dim} dim akcije, dobijeno {len(action)}"
        )

        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(dt)

        # update the live dashboard with camera observations produced by RoboSuite
        if "sideview_image" in obs:
            device.update_extra_frame("sideview", cv2.cvtColor(obs["sideview_image"], cv2.COLOR_RGB2BGR))
        if "robot0_eye_in_hand_image" in obs:
            device.update_extra_frame("robot0_eye_in_hand", cv2.cvtColor(obs["robot0_eye_in_hand_image"], cv2.COLOR_RGB2BGR))
        if "agentview_image" in obs:
            device.update_extra_frame("agentview", cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR))

        step_count += 1
        success = hanoi_env._check_success()
        legal, illegal_reason = hanoi_env.check_hanoi_legality()
        if not legal:
            episode_always_legal = False


        device.update_status([
            f"[RECORDING] step {step_count}",
            f"Clutch: {'YES' if clutch_now else 'no'}",
            f"Grasp: {'YES' if device.grasp else 'no'}",
            f"EE: {solver.get_eef_position().round(3)}",
            "",
            "SOLVED! :)" if success else "in progress...",
            "Legal: YES" if legal else f"ILLEGAL: {illegal_reason}",
            f"(hold: {success_hold_count}/{SUCCESS_HOLD_STEPS})",
            f"Time remaining: {(env.horizon - step_count) * dt:.0f}s",
        ])

        
        if success:
            success_hold_count += 1
        else:
            success_hold_count = 0

        if success_hold_count >= SUCCESS_HOLD_STEPS:
            print(f"Demonstration completed (success held for {SUCCESS_HOLD_STEPS} steps).")

            break

        steps_remaining = env.horizon - step_count
        seconds_remaining = steps_remaining * dt

        # Warn the operator shortly before the demonstration reaches its maximum duration
        if steps_remaining in (int(30 / dt), int(10 / dt)):
            print(f"WARNING: approximately {seconds_remaining:.0f}s remaining before this demonstration times out.")

        if step_count >= env.horizon:
            print("Demonstration timed out without reaching the success condition and will not be stored.")
            break

    if not episode_always_legal:
        print("WARNING: this episode violated a Hanoi rule at least once during execution. The detailed legality history can be inspected using validate_dataset.py.")

    return episode_always_legal


if __name__ == "__main__":
    env, env_info = build_env()

    # IK configuration passed to collect_one_demonstration().
    # The solver itself is instantiated after each environment reset so that
    # it always references the current simulation object.
    ik_kwargs = dict(arm="right", damping=0.05, step_size=0.5, max_joint_step=0.2)
    device = MediaPipeDevice(env=env, model_path="hand_landmarker.task")

    dt = 1.0 / env.control_freq

    # Wrap the environment with DataCollectionWrapper so that states,
    # actions, and success information are recorded during execution.
    #
    # The temporary directory is session-specific. Its contents are merged
    # into the persistent HDF5 dataset after each completed demonstration
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # Persistent dataset location. All recording sessions append to the same
    # HDF5 file, allowing demonstrations to accumulate over time.
    new_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demonstrations", "hanoi_dataset")
    os.makedirs(new_dir, exist_ok=True)

    print(f"Temporary recordings: {tmp_directory}")
    print(f"Final HDF5 dataset: {new_dir}/demo.hdf5")
    print("Press Ctrl+C in the terminal to stop the recording session at any time.\n")

    try:
        while True:
            collect_one_demonstration(env, device, ik_kwargs, dt)

            # DataCollectionWrapper buffers recorded trajectory data.
            # Flush it explicitly so that the demonstration that has just
            # finished is available to the HDF5 conversion step immediately.
            env._flush()

            # Merge the latest recorded episode into the persistent dataset.
            # Performing this after every demonstration ensures that already
            # collected data is preserved even if the session is interrupted.
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)

    except KeyboardInterrupt:
        print("\nRecording interrupted -- the HDF5 dataset has already been updated after each completed demonstration.")
    finally:
        device.close()
        env.close()