"""
record_depth_session.py

Records a session for COMPARING DEPTH ESTIMATION METHODS.

Idea: record ONCE, analyze as many times as needed. The script stores, for
each frame: RGB image, aligned RealSense depth frame, detected MediaPipe
landmarks, and camera intrinsic parameters. All depth estimation methods
are later compared OFFLINE on the SAME frames (compare_depth_methods.py) --
without the camera, reproducibly, and fairly because every method receives
identical input.

RECORDING PROCEDURE:
    Mark several positions on the table with tape at known distances from
    the camera (default: 30, 40, 50, 60, 70 cm). The script guides you
    through them -- for each distance, hold your hand still until the
    required number of frames is recorded.

WHY KNOWN DISTANCES (instead of using only RealSense as reference):
    This provides TWO independent references -- the sensor and a ruler.
    If they agree, the reference is reliable; if not, you know that the
    sensor itself has an issue at that distance and can report it fairly
    instead of blindly trusting the sensor.

Usage:
    python record_depth_session.py
    python record_depth_session.py --distances 25 35 45 55
    python record_depth_session.py --frames 40
"""

import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_DISTANCES_CM = [30, 40, 50, 60, 70]
DEFAULT_FRAMES_PER_DISTANCE = 70
MODEL_PATH = "hand_landmarker.task"


def build_detector(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing {model_path}. Download it from the MediaPipe model zoo "
            "and place it in the same folder as this script."
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def start_realsense(width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # IMPORTANT: read the depth scale FROM THE DEVICE instead of assuming it.
    # D405 is a close-range camera and typically uses a finer unit (0.0001 m)
    # than the common 0.001 m. Hard-coding the wrong value causes a 10x error.
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale read from device: {depth_scale} m per unit")

    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    intrinsics = {
        "width": intr.width, "height": intr.height,
        "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
    }
    return pipeline, align, intrinsics, depth_scale


def capture_session(distances_cm, frames_per_distance, out_path):
    detector = build_detector()
    pipeline, align, intrinsics, depth_scale = start_realsense()

    print("=" * 60)
    print("RECORDING SESSION FOR DEPTH ESTIMATION COMPARISON")
    print("=" * 60)
    print(f"Distances: {distances_cm} cm")
    print(f"Frames per distance: {frames_per_distance}")
    print()
    print("For each distance: place your hand at the marked position, hold it")
    print("still and OPEN (palm facing the camera), then press ENTER in the terminal.")
    print("Press 'q' in the camera window to stop at any time.")
    print()

    color_frames, depth_frames, landmarks_all = [], [], []
    labels_cm, frame_ids = [], []

    try:
        for dist_cm in distances_cm:
            input(f">>> Place your hand at {dist_cm} cm and press ENTER...")
            print(f"    Recording {frames_per_distance} frames at {dist_cm} cm...", end="", flush=True)

            captured = 0
            attempts = 0
            while captured < frames_per_distance and attempts < frames_per_distance * 10:
                attempts += 1
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                    aligned = align.process(frames)
                    depth_frame = aligned.get_depth_frame()
                    color_frame = aligned.get_color_frame()
                except RuntimeError as e:
                    # Occasional frame failures should not terminate the session.
                    print(f"\n    [warning] skipping frame: {e}")
                    continue

                if not depth_frame or not color_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())          # BGR
                # .copy() is REQUIRED: np.asanyarray returns a VIEW into the
                # RealSense frame memory. Keeping that view occupies the buffer,
                # eventually causing align.process to fail after many frames..
                depth = np.asanyarray(depth_frame.get_data()).copy()   

                # Detection is performed on the RAW (unflipped) frame so that
                # pixel coordinates remain consistent with camera intrinsics.
                rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                if not result.hand_landmarks:
                    continue  

                lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)

                color_frames.append(cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 92])[1])
                depth_frames.append(depth)
                landmarks_all.append(lm)
                labels_cm.append(dist_cm)
                frame_ids.append(len(frame_ids))
                captured += 1

                # Flip ONLY for display; stored data remains unflipped.
                preview = cv2.flip(color.copy(), 1)
                h, w = preview.shape[:2]
                for p in result.hand_landmarks[0]:
                    cv2.circle(preview, (w - 1 - int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)
                cv2.putText(preview, f"{dist_cm} cm  [{captured}/{frames_per_distance}]",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Snimanje", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

            print(f" done ({captured} frames, {attempts} attempts)")

    except KeyboardInterrupt:
        print("\nInterrupted -- saving all frames recorded so far.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if not color_frames:
        print("No frames were recorded.")
        return None

    np.savez_compressed(
        out_path,
        color_jpg=np.array(color_frames, dtype=object),
        depth=np.array(depth_frames, dtype=np.uint16),
        landmarks=np.array(landmarks_all, dtype=np.float32),
        label_cm=np.array(labels_cm, dtype=np.int32),
        frame_id=np.array(frame_ids, dtype=np.int32),
        intrinsics=np.array([intrinsics], dtype=object),
        depth_scale=np.float32(depth_scale),
    )
    print(f"\nSaved {len(color_frames)} frames to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=DEFAULT_DISTANCES_CM,
                    help="known distances in cm")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES_PER_DISTANCE,
                    help="number of frames per distance")
    ap.add_argument("--out", type=str, default="sesija_dubina.npz")
    args = ap.parse_args()

    capture_session(args.distances, args.frames, args.out)