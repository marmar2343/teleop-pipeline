"""
test_teleoperation.py

Real-time hand-based teleoperation of a Panda robot using MediaPipe
hand tracking and RGB-D depth information from an Intel RealSense D405.

The hand displacement is converted into Cartesian end-effector velocity
using nonlinear rate control. The resulting target position is tracked
by a Damped Least Squares inverse kinematics solver.

The script also provides visualization of the simulated camera streams,
gripper state, end-effector position, hand displacement, and task status.
"""

import time

import cv2
import numpy as np
import robosuite as suite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

from mediapipe_device import MediaPipeDevice
from ik_solver import DLS_IK_Solver
import hanoi_three_env  # noqa: F401 


# CAMERA-TO-ROBOT AXIS MAPPING
# The hand displacement is expressed in the RealSense camera frame: X points right, Y points down, and Z points forward from the camera

SIGN_FORWARD = -1.0     # robot X (forward/backward) <- stick[2] (RealSense depth)
SIGN_LATERAL = 1.0    # robot Y (left/right)   <- stick[0] (RealSense X) 
SIGN_VERTICAL = -1.0   # robot Z (up/down)    <- stick[1] (RealSense Y)

# Cartesian velocity gains for the corresponding robot-frame directions
K_XY = 2.0  # m/s per meter of lateral or vertical displacement
K_Z = 2.0   # m/s per meter of depth displacement

# Safety limit on the magnitude of the Cartesian end-effector velocity.
MAX_LINEAR_SPEED = 0.3  # m/s


# NONLINEAR RESPONSE CURVE
# A nonlinear response provides finer control for small hand
# displacements while preserving a stronger response for larger intentional movements
CURVE_EXPONENT = 2.0   # 1.0 = linearno (bez efekta), 2.0 = kvadratno, veci broj = izrazenija razlika fino/brzo

# Expected maximum displacement after the deadzone.
# These values define the normalization range of the response curve.
STICK_MAX_XY = 0.15  # m
STICK_MAX_Z = 0.15   # m

def _apply_curve(value, max_expected, exponent):
    """
    Apply a nonlinear response curve while preserving the input sign.

    The magnitude is normalized to [0, 1], raised to the specified
    exponent, and mapped back to the original displacement scale.

    With exponent=2, a displacement equal to 50% of the expected
    maximum produces 25% of the output magnitude.
    """
    normalized = np.clip(abs(value) / max_expected, 0.0, 1.0)
    shaped = normalized ** exponent

    return np.sign(value) * shaped * max_expected


def build_env(control_freq=30, use_cameras=True):
    """
    Create and configure the Hanoi Towers robosuite environment.

    The Panda arm is controlled using absolute joint positions.
    Simulation cameras can optionally be enabled for visualization
    and data collection.
    """
    controller_config = load_composite_controller_config(controller="BASIC")
    controller_config["body_parts"]["right"]["type"] = "JOINT_POSITION"
    controller_config["body_parts"]["right"]["input_type"] = "absolute"
    controller_config["body_parts"]["right"]["interpolation"] = "linear"

    env = suite.make(
        env_name="HanoiThree",
        robots="Panda",
        controller_configs=controller_config,
        source_peg_idx=0,
        target_peg_idx=2,
        randomize_pegs=False,
        color_code_pegs=True,  
        has_renderer=True,     
        has_offscreen_renderer=use_cameras,  
        use_camera_obs=use_cameras,
        camera_names=["sideview", "robot0_eye_in_hand", "agentview"] if use_cameras else None,
        camera_heights=320 if use_cameras else None,
        camera_widths=320 if use_cameras else None,
        control_freq=control_freq,
        horizon=2000,
        ignore_done=True,
    )
    return VisualizationWrapper(env)


def stick_to_velocity(stick):
    """
    Convert hand displacement into Cartesian end-effector velocity.

    Args:
        stick: 3D hand displacement in the RealSense camera frame.

    Returns:
        velocity: Desired end-effector linear velocity in the robot
            frame, expressed in m/s.

    The camera-frame displacement is first shaped by the nonlinear
    response curve. The resulting components are then mapped to the
    robot coordinate system using the configured axis signs and
    converted to velocity using the corresponding gains.
    """

    shaped_x = _apply_curve(stick[0], STICK_MAX_XY, CURVE_EXPONENT)
    shaped_y = _apply_curve(stick[1], STICK_MAX_XY, CURVE_EXPONENT)
    shaped_z = _apply_curve(stick[2], STICK_MAX_Z, CURVE_EXPONENT)

    velocity = np.array([
        shaped_z * K_Z * SIGN_FORWARD,
        shaped_x * K_XY * SIGN_LATERAL,
        shaped_y * K_XY * SIGN_VERTICAL,])

    speed = np.linalg.norm(velocity)
    if speed > MAX_LINEAR_SPEED:
        velocity = velocity * (MAX_LINEAR_SPEED / speed)

    return velocity


if __name__ == "__main__":
    env = build_env()
    obs = env.reset()
    env.render()
    env.robots[0].print_action_info()

    solver = DLS_IK_Solver(env, arm="right", damping=0.05, step_size=0.5, max_joint_step=0.2)

    device = MediaPipeDevice(env=env, model_path="hand_landmarker.task")
    device.start_control()

    dt = 1.0 / env.control_freq

    # initialize the target from the actual end-effector position
    target_ee_pos = solver.get_eef_position()
    prev_clutch = False
    step_count = 0

    print("Started (rate control). Hold SPACE to control the robot with your hand. Press Ctrl+C in the terminal to stop.")

    try:
        while True:
            # Read the hand-tracking state while holding the device lock
            # to avoid accessing partially updated values from another thread.
            with device.lock:
                clutch_now = device._clutch_active
                stick = device.stick.copy()

            q_current = solver.get_current_qpos()

            if clutch_now and not prev_clutch:
                # Synchronize the Cartesian target with the actual robot
                # position when the clutch is engaged. This prevents a
                # discontinuity between separate teleoperation intervals.

                target_ee_pos = solver.get_eef_position()

            if clutch_now:
                # Rate control: hand displacement determines velocity,
                # which is integrated to obtain the next Cartesian target.
                velocity = stick_to_velocity(stick)
                target_ee_pos = target_ee_pos + velocity * dt

                # convert the Cartesian target into joint-space commands
                q_target = solver.step(q_current, target_ee_pos)
            else:
                # Keep the robot at its current configuration while the
                # clutch is released. The Cartesian target is not updated
                q_target = q_current

            prev_clutch = clutch_now

            gripper_action = np.array([1.0 if device.grasp else -1.0])
            action = np.concatenate([q_target, gripper_action])

            assert len(action) == env.action_dim, (
                f"Expected action dimension {env.action_dim}, "
                f"got {len(action)} - "
                f"check the number of arm joints / gripper dimensions")

            obs, reward, done, info = env.step(action)
            env.render()

            time.sleep(dt)

            # Forward simulated camera frames to the visualization device. RGB images from robosuite are converted to OpenCV's BGR format.
            if "sideview_image" in obs:
                device.update_extra_frame("sideview", cv2.cvtColor(obs["sideview_image"], cv2.COLOR_RGB2BGR))
            if "robot0_eye_in_hand_image" in obs:
                device.update_extra_frame("robot0_eye_in_hand", cv2.cvtColor(obs["robot0_eye_in_hand_image"], cv2.COLOR_RGB2BGR))
            if "agentview_image" in obs:
                device.update_extra_frame("agentview", cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR))

            step_count += 1

            # evaluate task completion and Hanoi-specific legality
            success = env.env._check_success()
            legal, illegal_reason = env.env.check_hanoi_legality()

            # update the visualization panel with the current teleoperation and task state
            ee_now = solver.get_eef_position()
            device.update_status([
                f"Clutch: {'YES' if clutch_now else 'NO'}",
                f"Grasp: {'YES' if device.grasp else 'NO'}",
                f"EE: {ee_now.round(3)}",
                f"Stick: {stick.round(3)}",
                "",
                "Solved! :)" if success else "in progress...",
                "LEGAL: YES" if legal else f"NOT LEGAL: {illegal_reason}",
            ])

            if step_count % 10 == 0:
                status = "SOLVED!" if success else "in progress"
                legal_str = "OK" if legal else f"NOT LEGAL ({illegal_reason})"
                print(f"clutch={clutch_now}, stick={stick.round(3)}, ee={ee_now.round(3)}, status={status}, legality={legal_str}")

            if done:
                print("Episode finished, reseting...")
                obs = env.reset()
                target_ee_pos = solver.get_eef_position()

    except KeyboardInterrupt:
        print("Interrupting.")
    finally:
        device.close()
        env.close()