import threading
import numpy as np
import cv2
import mediapipe as mp
import time
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
        deadzone_xy=0.015,
        deadzone_z=0.008,
        stick_alpha=0.25,
    ):
        super().__init__(env)

        self.GRASP_THRESHOLD = 0.04
        self.deadzone_xy = deadzone_xy
        self.deadzone_z = deadzone_z
        self.stick_alpha = stick_alpha

        # MediaPipe detector
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame_timestamp_ms = 0

        # Performance monitoring
        self.camera_times = deque(maxlen=30)
        self.mediapipe_times = deque(maxlen=30)
        self.control_times = deque(maxlen=30)
        self.last_perf_print = time.perf_counter()
        self.last_control_print = time.perf_counter()

        # Shared state
        self.lock = threading.Lock()
        self._clutch_active = False
        self.current_wrist_image = None
        self.current_wrist_world_z = None
        self.reference_wrist_image = None
        self.reference_wrist_world_z = None
        self.stick = np.zeros(3)

        self._enabled = False
        self._running = True

        # Robosuite states
        self._reset_internal_state()
        self._reset_state = 0

        # Keyboard listener
        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

        # Camera thread
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()

        print("SPACE hold = tracking (rate control, 2.5D)")

    def _reset_internal_state(self):
        super()._reset_internal_state()
        self.raw_drotation = np.zeros(3)
        self.last_drotation = np.zeros(3)
        self.last_stick = np.zeros(3)
        self.stick = np.zeros(3)

    def start_control(self):
        self._reset_internal_state()
        self._enabled = True

    @staticmethod
    def _deadzone(value, dz):
        if abs(value) < dz:
            return 0.0
        return value - dz * np.sign(value)

    def _on_press(self, key):
        if key == Key.space:
            with self.lock:
                if not self._clutch_active:
                    self._clutch_active = True
                    if self.current_wrist_image is not None and self.current_wrist_world_z is not None:
                        self.reference_wrist_image = self.current_wrist_image.copy()
                        self.reference_wrist_world_z = self.current_wrist_world_z
                        self.stick = np.zeros(3)

    def _on_release(self, key):
        if key == Key.space:
            with self.lock:
                self._clutch_active = False

    def get_controller_state(self):
        t0 = time.perf_counter()

        with self.lock:
            current_stick = self.stick.copy()

        dstick = current_stick - self.last_stick
        self.last_stick = current_stick

        raw_drotation = self.raw_drotation - self.last_drotation
        self.last_drotation = self.raw_drotation.copy()

        t1 = time.perf_counter()
        self.control_times.append(t1 - t0)

        if time.perf_counter() - self.last_control_print > 1.0 and self.control_times:
            controller_hz = 1 / np.mean(self.control_times)
            print(f"Controller frequency: {controller_hz:.1f} Hz")
            self.last_control_print = time.perf_counter()

        return dict(
            dpos=dstick,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def _camera_loop(self):
        while self._running:
            loop_start = time.perf_counter()
            ret, frame = self.cap.read()
            camera_end = time.perf_counter()

            if not ret:
                continue

            self.camera_times.append(camera_end - loop_start)

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.frame_timestamp_ms += 33

            mp_start = time.perf_counter()
            result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
            mp_end = time.perf_counter()
            self.mediapipe_times.append(mp_end - mp_start)

            if result.hand_world_landmarks and result.hand_landmarks:
                world_landmarks = result.hand_world_landmarks[0]
                image_landmarks = result.hand_landmarks[0]

                wrist_world = world_landmarks[0]
                wrist_image = image_landmarks[0]

                # Grasp detection via pinch distance
                thumb_tip = world_landmarks[4]
                index_tip = world_landmarks[8]
                distance = np.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2 +
                    (thumb_tip.z - index_tip.z) ** 2
                )
                is_grasping = distance < self.GRASP_THRESHOLD

                with self.lock:
                    self.grasp_states[self.active_robot][self.active_arm_index] = is_grasping

                wrist_image_pos = np.array([wrist_image.x, wrist_image.y])
                wrist_world_z = wrist_world.z

                with self.lock:
                    self.current_wrist_image = wrist_image_pos.copy()
                    self.current_wrist_world_z = wrist_world_z

                    if self._enabled and self._clutch_active and self.reference_wrist_image is not None:
                        offset_x = wrist_image_pos[0] - self.reference_wrist_image[0]
                        offset_y = wrist_image_pos[1] - self.reference_wrist_image[1]
                        offset_z = wrist_world_z - self.reference_wrist_world_z

                        offset_x = self._deadzone(offset_x, self.deadzone_xy)
                        offset_y = self._deadzone(offset_y, self.deadzone_xy)
                        offset_z = self._deadzone(offset_z, self.deadzone_z)

                        raw_stick = np.array([offset_x, offset_y, offset_z])
                        self.stick = self.stick_alpha * raw_stick + (1 - self.stick_alpha) * self.stick

            # Performance logging
            if time.perf_counter() - self.last_perf_print > 1.0:
                camera_fps = 1 / np.mean(self.camera_times)
                mp_latency = np.mean(self.mediapipe_times) * 1000
                print(f"\n----- PERFORMANCE -----\nCamera FPS: {camera_fps:.1f}\nMediaPipe latency: {mp_latency:.1f} ms\n-----------------------\n")
                self.last_perf_print = time.perf_counter()

            # Display frame and info
            with self.lock:
                clutch = self._clutch_active
                stick_display = self.stick.copy()

            text = "TRACKING" if clutch else "PAUSED"
            color = (0, 255, 0) if clutch else (0, 0, 255)

            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if clutch:
                cv2.putText(frame, f"stick: {stick_display.round(3)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if result.hand_landmarks:
                h, w, _ = frame.shape
                for lm in result.hand_landmarks[0]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.imshow("MediaPipe Teleoperation", frame)
            cv2.waitKey(1)

    def close(self):
        self._running = False
        self.thread.join(timeout=2)
        self.cap.release()
        cv2.destroyAllWindows()
        self.listener.stop()