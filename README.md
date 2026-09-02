# Markerless RGB-D Teleoperation Pipeline

A markerless teleoperation system for a Panda robotic manipulator, based on an Intel RealSense D405 RGB-D camera and MediaPipe hand detection.

The system was tested on the Tower of Hanoi task with 3 cubes, with demonstrations recorded in a format suitable for imitation learning and VLA models.

## Repository Structure

```text
pipeline/           Core system -- detection, IK, environment, recording
data_collection/    Tools for dataset inspection and statistics
experiments/        Controller comparison, depth estimation, analysis
requirements.txt    Required Python packages
```

### `pipeline/`

| File                    | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| `mediapipe_device.py`   | RealSense + MediaPipe hand detection, rate control and dashboard              |
| `ik_solver.py`          | `DLS_IK_Solver` -- differential inverse kinematics using damped least squares |
| `hanoi_three_env.py`    | `HanoiThree` -- custom RoboSuite environment for the task                     |
| `run_teleop.py`         | Main script for recording demonstrations to `demo.hdf5`                       |
| `test_teleoperation.py` | Teleoperation test without recording, used for controller tuning              |
| `hand_landmarker.task`  | **Download manually** -- see the Installation section below                   |

### `data_collection/`

| File                    | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| `inspect_dataset.py`    | Inspect `demo.hdf5`, replay episodes and export videos                    |
| `dataset_statistics.py` | Dataset statistics: success rate, episode length and gripper trajectories |

### `experiments/`

| File                            | Description                                                                                              |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `compare_controllers.py`        | Comparison of the custom IK solver with the RoboSuite `OSC_POSE` controller                              |
| `plot_controller_comparison.py` | Plots and LaTeX table for controller comparison                                                          |
| `record_depth_session.py`       | Records a session for comparing depth estimation methods (requires camera)                               |
| `compare_depth_methods.py`      | Offline analysis of a single session (RealSense / YOLO26-depth / Depth Anything V2 / apparent hand size) |
| `merge_depth_sessions.py`       | Merges multiple sessions and evaluates cross-session repeatability                                       |
| `plot_depth_comparison.py`      | Additional plots and LaTeX table for depth estimation results                                            |

## Installation

```bash
pip install -r requirements.txt
```

**MuJoCo version note:** `robosuite==1.5.2` is not compatible with `mujoco>=3.4` and raises a `TypeError` in `mj_fullM`. Therefore, `requirements.txt` explicitly requires `mujoco==3.3.0`.

### `hand_landmarker.task` -- required manual step

The MediaPipe hand detection model is not installed through `pip`. It must be downloaded separately as a model file. Without it, `mediapipe_device.py` will fail at startup.

1. Download `hand_landmarker.task` from the official MediaPipe model page.
   Use the Hand Landmarker **float16** or **full** model variant.
2. Place the file directly inside the `pipeline/` folder, at the same level as `run_teleop.py`.

## Running the System

### Recording demonstrations

A RealSense camera is required.

```bash
cd pipeline
python run_teleop.py
```

Hold **SPACE** to enable hand tracking. A pinch gesture (thumb + index finger) closes the gripper.

Demonstrations are saved to:

```text
pipeline/demonstrations/hanoi_dataset/demo.hdf5
```

New demonstrations are appended to the existing dataset across recording sessions.

### Inspecting the dataset

```bash
cd data_collection

python inspect_dataset.py ../pipeline/demonstrations/hanoi_dataset/demo.hdf5

python dataset_statistics.py ../pipeline/demonstrations/hanoi_dataset/demo.hdf5
```

### Controller comparison

No camera is required. The comparison uses a scripted trajectory.

```bash
cd experiments

python -c "import robosuite.macros as m; m.IMAGE_CONVENTION='opencv'; from compare_controllers import run_full_comparison; run_full_comparison(n_repeats=5)"

python plot_controller_comparison.py
```

The comparison evaluates the custom DLS inverse kinematics solver against the RoboSuite `OSC_POSE` controller.

### Depth estimation evaluation

A RealSense camera is required for recording the depth sessions. Additional libraries are required for the monocular depth estimation methods.

```bash
cd experiments

pip install ultralytics transformers torch
```

Record a session:

```bash
python record_depth_session.py --out session_1.npz
```

Analyze the recorded session offline:

```bash
python compare_depth_methods.py --session session_1.npz --depth-scale 0.0001 --out result_1.json
```

Repeat the recording and analysis for additional sessions:

```text
session_2.npz -> result_2.json
session_3.npz -> result_3.json
...
```

Merge multiple sessions to evaluate cross-session repeatability:

```bash
python merge_depth_sessions.py result_1.json result_2.json result_3.json
```

Generate plots and the LaTeX table for a single depth-estimation session:

```bash
python plot_depth_comparison.py --results result_1.json
```

The depth evaluation compares all methods at the **same image pixel**, corresponding to the detected wrist landmark (MediaPipe landmark 0). The RealSense measurement and ruler measurements provide independent references.

## Known Limitations

* The `IK_POSE` controller in `robosuite==1.5.2` has a bug where the reference pose used by the null-space term is not updated between calls. This causes drift even when the commanded pose remains unchanged. Therefore, the controller comparison uses `OSC_POSE` as the RoboSuite reference.
* The monocular depth methods do not necessarily provide a perfectly calibrated absolute depth scale. For this reason, both raw and scale-aligned results are evaluated where applicable.
* The `mp_size` method is a baseline based on the apparent size of the hand in the image. It provides a relative depth cue rather than a direct metric measurement.
* Depth estimation is evaluated offline on previously recorded RGB-D sessions, so the camera is not required during the comparison or plotting stages.
