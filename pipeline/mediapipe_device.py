"""
mediapipe_device.py
 
MediaPipe hand-tracking teleoperation device for robosuite, using rate
control and an Intel RealSense D405 depth camera.
 
Hand position and pinch (grasp) distance are both computed via true
depth deprojection: MediaPipe locates 2D keypoints in the image, and the
RealSense depth stream supplies the measured distance at each keypoint's
pixel. MediaPipe's own metric estimate (hand_world_landmarks) is not used
for any measurement - its absolute error per fingertip is roughly
constant regardless of hand distance, which is negligible for the wide
spacing of an open hand but dominant for the small pinch distance between
thumb and index finger, so relying on it would make the grasp threshold
distance-dependent.
 
Detection and deprojection run on the raw (unflipped) camera frame,
since rs2_deproject_pixel_to_point requires pixel coordinates consistent
with the sensor's actual intrinsics. The frame is flipped only on a
separate copy used for the cv2.imshow preview, so the operator sees a
natural mirror image without affecting any measurement.
 
The device does not compute an absolute end-effector target. self.stick
is a "joystick" offset from a reference point, intended for rate control;
the velocity mapping and gain live in test_teleoperation.py / run_teleop.py.
 
RealSense camera-frame convention (looking from the camera into the
scene): X right, Y down, Z forward (depth, increasing away from camera).
 
SPACE clutch:
- hold SPACE   -> robot follows the hand's offset from the reference point
                  (velocity is integrated)
- release SPACE -> robot holds its current position
"""

import threading
import numpy as np
import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pynput.keyboard import Key, Listener

from robosuite.devices import Device


class MediaPipeDevice(Device):

    def __init__(
        self,
        env,
        model_path="hand_landmarker.task",
        deadzone_xy=0.006,   # in meters, for lateral/ vertical movement, dead zone is used to filter out noise
        deadzone_z=0.006,    # meters, for depth (RealSense depth)
        stick_alpha=0.25,    # EMA smoothing applied after the dead zone
        grasp_close_threshold=0.035,  # below this: open->closed
        grasp_open_threshold=0.05,    # above this: closed->open - the gap between the two is the HYSTERESIS band
        grasp_alpha=0.4,               # EMA smoothing applied to the raw pinch distance, before the open/closed decision
    ):

        super().__init__(env)

        self.GRASP_CLOSE_THRESHOLD = grasp_close_threshold
        self.GRASP_OPEN_THRESHOLD = grasp_open_threshold
        self.grasp_alpha = grasp_alpha
        self._pinch_distance_smooth = grasp_open_threshold  # start in the open position

        self.deadzone_xy = deadzone_xy
        self.deadzone_z = deadzone_z
        self.stick_alpha = stick_alpha

        # MediaPipe init

        base_options = python.BaseOptions(
            model_asset_path=model_path # path to the hand_landmarker.task
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)


        # Camera

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.rs_profile = self.rs_pipeline.start(rs_config)

        # aligns the depth frame to the color frame, because two sensors are physically offset on the device
        self.rs_align = rs.align(rs.stream.color)

        # intrinsics of the COLOR stream - needed to deproject pixel+depth into a 3D point
        color_profile = self.rs_profile.get_stream(rs.stream.color)
        self.rs_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        self.frame_timestamp_ms = 0
        self._camera_start_time = time.perf_counter()


        # Performance monitoring


        self.camera_times = deque(maxlen=30)
        self.mediapipe_times = deque(maxlen=30)
        self.control_times = deque(maxlen=30)

        self.last_perf_print = time.perf_counter()
        self.last_control_print = time.perf_counter()


        # Shared state

        self.lock = threading.Lock()
        self._clutch_active = False

        # Images that an external caller (e.g. the main loop in test_teleoperationn.py)
        # wants shown via cv2.imshow are written here rather than displayed directly, because
        # OpenCV's GUI calls are not thread-safe,  so only the camera thread (_camera_loop) is allowed to call cv2.imshow
        # other threads just hand it the image through this dict. Key = window name.
        self.extra_frames = {}
        self._known_extra_windows = set()
        self.status_lines = []

        # Current (updated every frame) 3D position of the wrist keypoint in the camera frame (meters), 
        # obtained via RealSense deprojection (pixel + measured depth -> 3D point).
        self.current_wrist_3d = None

        # Reference (neutral) position, captured at the moment SPACE is pressed
        self.reference_wrist_3d = None
        self.stick = np.zeros(3)
        self._enabled = False
        self._running = True


        # Robosuite states

        self._reset_internal_state()
        self._reset_state = 0


        # Keyboard

        self.listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )

        self.listener.start()


        # Camera thread

        self.thread = threading.Thread(
            target=self._camera_loop, daemon=True
        )

        self.thread.start()

        print("Hold SPACE to track (rate control, 3D)")



    def _reset_internal_state(self):

        super()._reset_internal_state()
        self.raw_drotation = np.zeros(3)
        self.last_drotation = np.zeros(3)
        self.last_stick = np.zeros(3)
        self.stick = np.zeros(3)



    def start_control(self):

        self._reset_internal_state()
        self._enabled = True



    def update_extra_frame(self, name, image_bgr):
        """
        Thread-safely hand an image for the camera thread to display in a cv2 window.
        Call this from any other thread. Never call cv2.imshow from another thread.
        """
        with self.lock:
            self.extra_frames[name] = image_bgr



    def update_status(self, lines):
        """
        Thread-safely hand off a list of strings to show on the dashboard's status panel.
        (e.g. end-effector position, task status only the main loop knows about).
        Same principle as update_extra_frame
        """
        with self.lock:
            self.status_lines = list(lines)



    @staticmethod
    def _deadzone(value, dz):
        """
        Dead zone: values within [-dz, dz] become 0; outside that range the output continues smoothly from 0 
        (dz is subtracted rather than the value simply clipped, to avoid a jump at the zone boundary).
        """
        if abs(value) < dz:
            return 0.0
        return value - dz * np.sign(value)



    # keyboard

    def _on_press(self,key):

        if key == Key.space:
            with self.lock:
                if not self._clutch_active:

                    self._clutch_active = True
                    if self.current_wrist_3d is not None:

                        self.reference_wrist_3d = (self.current_wrist_3d.copy()) 
                        self.stick = np.zeros(3)  # start each new tracking session from a clean slate


    def _on_release(self,key):

        if key == Key.space:

            with self.lock:

                self._clutch_active = False



    """"
    controller interface (satisfies robosuite's Device base class/input2action path, 
    which this pipeline does not otherwise use. kept so nothing breaks if it is called, 
    but do not rely on it for the rate-control logic used elsewhere in this project)
    """

    def get_controller_state(self):

        t0 = time.perf_counter()

        with self.lock:
            current_stick = self.stick.copy()

        dstick = current_stick - self.last_stick
        self.last_stick = current_stick

        raw_drotation = (self.raw_drotation - self.last_drotation)
        self.last_drotation = (self.raw_drotation.copy())

        t1 = time.perf_counter()
        self.control_times.append(t1 - t0)

        if time.perf_counter() - self.last_control_print > 1.0:

            if len(self.control_times):

                controller_hz = (1 / np.mean(self.control_times))
                print(f"Controller frequency: {controller_hz:.1f} Hz")

            self.last_control_print = time.perf_counter()


        return dict( dpos=dstick, rotation=self.rotation, raw_drotation=raw_drotation,
            grasp=int(self.grasp), reset=self._reset_state, base_mode=int(self.base_mode),)



    # camera thread

    def _camera_loop(self):

        while self._running:
            loop_start = time.perf_counter()
        
            # camera read timing
            rs_frames = self.rs_pipeline.wait_for_frames()
            aligned = self.rs_align.process(rs_frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            camera_end = time.perf_counter()


            if not depth_frame or not color_frame:
                continue


            self.camera_times.append(camera_end-loop_start)
            frame = np.asanyarray(color_frame.get_data())

           # IMPORTANT: do not flip here -- MediaPipe detection and
            # deprojection (rs2_deproject_pixel_to_point) must operate on
            # the raw pixel coordinates, consistent with the sensor's
            # actual intrinsics. Flipping is done only at the end, on a
            # copy, purely for the cv2.imshow preview.
 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Use actual elapsed time rather than a fixed per-frame increment
            self.frame_timestamp_ms = int((time.perf_counter() - self._camera_start_time) * 1000)

            # MediaPipe timing
            mp_start = time.perf_counter()
            result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
            mp_end = time.perf_counter()

            self.mediapipe_times.append(mp_end-mp_start)


            # Hand processing

            if result.hand_world_landmarks and result.hand_landmarks:


                # hand_world_landmarks is not used for any measurement (neither pinch distance nor position)
                image_landmarks = result.hand_landmarks[0]
                wrist_image = image_landmarks[0]

                h, w = frame.shape[:2]

                # -------------------------
                # GRASP DETECTION - pinch distance is measured via true RealSense depth at both 
                # fingertips (same approach used for the wrist position below).
                #
                # A model-based metric estimate of fingertip position
                # (such as hand_world_landmarks) carries an absolute error
                # that is roughly constant in millimeters regardless of
                # hand distance. That error is negligible next to the wide
                # spacing of an open hand (~8 cm) but becomes dominant
                # next to the small pinch distance itself, which would
                # make the measured distance vary noticeably with distance
                # from the camera. Measuring true depth at each fingertip
                # avoids this.
                #
                # EMA smoothing plus hysteresis (two thresholds) below
                # still guard against ordinary sensor noise on top of this
                # more reliable raw signal.
                # -------------------------

                thumb_tip_image = image_landmarks[4]
                index_tip_image = image_landmarks[8]

                thumb_px = int(np.clip(thumb_tip_image.x * w, 0, w - 1))
                thumb_py = int(np.clip(thumb_tip_image.y * h, 0, h - 1))
                index_px = int(np.clip(index_tip_image.x * w, 0, w - 1))
                index_py = int(np.clip(index_tip_image.y * h, 0, h - 1))

                thumb_depth_m = depth_frame.get_distance(thumb_px, thumb_py)
                index_depth_m = depth_frame.get_distance(index_px, index_py)

                if thumb_depth_m > 0.0 and index_depth_m > 0.0:
                    thumb_3d = np.array(rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, [thumb_px, thumb_py], thumb_depth_m))
                    index_3d = np.array(rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, [index_px, index_py], index_depth_m))
                    raw_distance = np.linalg.norm(thumb_3d - index_3d)
                else:
                    # invalid depth at a fingertip (often happens during the pinch itself, when the 
                    # fingers partially occlude one another) - keep the last smoothed value
                    raw_distance = self._pinch_distance_smooth

                with self.lock:
                    self._pinch_distance_smooth = (self.grasp_alpha * raw_distance + (1 - self.grasp_alpha) * self._pinch_distance_smooth)
                    smoothed = self._pinch_distance_smooth
                    currently_grasping = self.grasp_states[self.active_robot][self.active_arm_index]

                    if currently_grasping:
                        is_grasping = smoothed < self.GRASP_OPEN_THRESHOLD
                    else:
                        is_grasping = smoothed < self.GRASP_CLOSE_THRESHOLD

                    self.grasp_states[self.active_robot][self.active_arm_index] = is_grasping


                wrist_image_pos = np.array([wrist_image.x, wrist_image.y])



                # DEPROJECTION: pixel + measured depth -> 3D point
                px = int(np.clip(wrist_image_pos[0] * w, 0, w - 1))
                py = int(np.clip(wrist_image_pos[1] * h, 0, h - 1))

                depth_m = depth_frame.get_distance(px, py)  # vec u metrima

                # depth_m == 0 means "no valid reading" at this pixel
                # (out of sensor range, IR reflection, shadow) - skip this frame
                if depth_m <= 0.0:
                    continue

                point_3d = rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, [px, py], depth_m)
                wrist_3d = np.array(point_3d)  # (X, Y, Z), camera frame


                with self.lock:

                    self.current_wrist_3d = wrist_3d.copy()
                    if (self._enabled and self._clutch_active and self.reference_wrist_3d is not None):

                        offset_x = wrist_3d[0] - self.reference_wrist_3d[0]
                        offset_y = wrist_3d[1] - self.reference_wrist_3d[1]
                        offset_z = wrist_3d[2] - self.reference_wrist_3d[2]

                        offset_x = self._deadzone(offset_x, self.deadzone_xy)
                        offset_y = self._deadzone(offset_y, self.deadzone_xy)
                        offset_z = self._deadzone(offset_z, self.deadzone_z)

                        raw_stick = np.array([offset_x, offset_y, offset_z])

                        self.stick = (self.stick_alpha * raw_stick + (1 - self.stick_alpha) * self.stick)

            # performance print

            if time.perf_counter() - self.last_perf_print > 1.0:

                camera_fps = (1 /np.mean(self.camera_times))

                mp_latency = (np.mean(self.mediapipe_times)*1000)


                print("\n----- PERFORMANCE -----")
                print(f"Camera FPS: {camera_fps:.1f}")
                print(f"MediaPipe latency: {mp_latency:.1f} ms")
                print("-----------------------\n")

                self.last_perf_print = time.perf_counter()


            # display 
            with self.lock:
                clutch = self._clutch_active
                stick_display = self.stick.copy()
                pinch_display = self._pinch_distance_smooth
                extra_frames_copy = dict(self.extra_frames)
                status_lines_copy = list(self.status_lines)

            device_status_lines = [
                f"Tracking: {'YES' if clutch else 'NO'}",
                f"Pinch (smooth): {pinch_display:.4f}",
                f"  close<{self.GRASP_CLOSE_THRESHOLD:.3f}  open>{self.GRASP_OPEN_THRESHOLD:.3f}",
                f"Stick: {stick_display.round(3)}" if clutch else "Stick: (clutch not active)",
                "",
            ]

            # camera preview: flip first, then draw landmark dots with a mirrored X coordinate so they
            # line up with the flipped image
            real_cam_frame = cv2.flip(frame, 1)
            if result.hand_landmarks:
                rh, rw = real_cam_frame.shape[:2]
                for lm in result.hand_landmarks[0]:
                    x = rw - 1 - int(lm.x * rw)
                    y = int(lm.y * rh)
                    cv2.circle(real_cam_frame, (x, y), 3, (0, 255, 0), -1)

            dashboard = self._build_dashboard(real_cam_frame, extra_frames_copy, device_status_lines + status_lines_copy)

            if "dashboard" not in self._known_extra_windows:
                cv2.namedWindow("Teleoperation - Dashboard", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Teleoperation - Dashboard", 900, 700)
                self._known_extra_windows.add("dashboard")

            cv2.imshow("Teleoperation - Dashboard", dashboard)


            cv2.waitKey(1)



    def _build_dashboard(self, real_camera_frame, extra_frames, status_lines, tile_size=320):
        """
        Combines the real camera (RealSense, with hand-detection dots) +
        agentview + robot0_eye_in_hand (simulation cameras) + a text
        status panel into a single image, in a 2x2 layout.
        """
        def prep(img, title):
            if img is None:
                tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)

            cv2.rectangle(tile, (0, 0), (tile_size, 22), (40, 40, 40), -1)
            cv2.putText(tile, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            return tile

        real_tile = prep(real_camera_frame, "Camera (hand)")
        agent_tile = prep(extra_frames.get("agentview"), "Agentview camera (sim)")
        eye_tile = prep(extra_frames.get("robot0_eye_in_hand"), "Wrist camera (sim)")

        status_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        cv2.rectangle(status_tile, (0, 0), (tile_size, 22), (40, 40, 40), -1)
        cv2.putText(status_tile, "Status", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for i, line in enumerate(status_lines):
            cv2.putText(
                status_tile, str(line), (10, 45 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        top = np.hstack([real_tile, agent_tile])
        bottom = np.hstack([eye_tile, status_tile])
        return np.vstack([top, bottom])
    

    def close(self):

        self._running=False
        self.thread.join(timeout=2)
        self.rs_pipeline.stop()
        cv2.destroyAllWindows()
        self.listener.stop()