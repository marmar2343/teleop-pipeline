"""
mediapipe_device.py

MediaPipe hand teleoperation device za robosuite -- v4, RATE CONTROL + RealSense D405.

PROMENA U ODNOSU NA v3 (2.5D: slika X/Y + MediaPipe world-landmark Z):

  - X/Y/Z SADA sve tri dolaze iz PRAVE dubinske deprojekcije: MediaPipe
    detektuje piksel poziciju zgloba sake na slici (2D, kao i pre), a
    RealSense depth frejm daje IZMERENU dubinu na tom pikselu -- pa
    rs2_deproject_pixel_to_point pretvara (piksel, dubina) u pravu 3D
    tacku u kamera frame-u (metri). Vise NEMA mesanja jedinica -- sve tri
    komponente self.stick su sada u istim, PRAVIM metarskim jedinicama.
  - hand_world_landmarks (MediaPipe naucena metricka procena) se i dalje
    koristi, ali SAMO za grasp detekciju (relativno rastojanje palac-
    kaziprst) -- ne vise za apsolutnu poziciju, pa mu tacnost apsolutne
    dubine tu i nije bitna.
  - VAZNA SUPTILNOST: MediaPipe detekcija i deprojekcija rade na SIROVOM
    (neflipovanom) frejmu -- rs2_deproject_pixel_to_point zahteva piksel
    koordinate koje odgovaraju stvarnim intrinsics-ima senzora. Flip
    (ogledalo) se radi TEK na kraju, na kopiji, samo za cv2.imshow prikaz.
    Sa v3 (bez prave deprojekcije) ovo nije bilo bitno jer se eventualna
    greska apsorbovala u SIGN_LATERAL konstantu -- sa pravom deprojekcijom
    bi flip pre detekcije davao POGRESNU X koordinatu.
  - Device i dalje NE racuna apsolutnu poziciju hvataljke -- self.stick je
    i dalje "otklon dzojstika" od referentne tacke, za rate control.
    Mapiranje/pojacanje ostaje u test_teleoperation.py.

RealSense konvencija kamera-frame-a (gledano IZ kamere KA sceni): X desno,
Y dole, Z napred (dubina, dalje od kamere).

SPACE clutch:
- hold SPACE -> robot prati otklon ruke od referentne tacke (integrise brzinu)
- release SPACE -> robot miruje na trenutnoj poziciji
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
        deadzone_xy=0.006,   # metri -- lateralno/vertikalno (RealSense X/Y), realna dubina je preciznija od MediaPipe procene pa moze manja mrtva zona
        deadzone_z=0.006,    # metri -- dubina (RealSense Z)
        stick_alpha=0.25,    # EMA glacanje NAD dead-zoned vrednoscu
    ):

        super().__init__(env)

        self.GRASP_THRESHOLD = 0.05

        self.deadzone_xy = deadzone_xy
        self.deadzone_z = deadzone_z
        self.stick_alpha = stick_alpha

        # -------------------------
        # MediaPipe
        # -------------------------

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)


        # -------------------------
        # Camera
        # -------------------------

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.rs_profile = self.rs_pipeline.start(rs_config)

        # poravnava depth frejm na color frejm -- bez ovoga depth i color
        # piksel (x,y) NE gledaju u istu tacku scene (kamere su fizicki
        # razmaknute na senzoru)
        self.rs_align = rs.align(rs.stream.color)

        # intrinsics COLOR streama -- potrebni za deprojekciju piksel+dubina -> 3D tacka
        color_profile = self.rs_profile.get_stream(rs.stream.color)
        self.rs_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        self.frame_timestamp_ms = 0



        # -------------------------
        # Performance monitoring
        # -------------------------

        self.camera_times = deque(maxlen=30)
        self.mediapipe_times = deque(maxlen=30)
        self.control_times = deque(maxlen=30)

        self.last_perf_print = time.perf_counter()
        self.last_control_print = time.perf_counter()



        # -------------------------
        # Shared state
        # -------------------------

        self.lock = threading.Lock()


        self._clutch_active = False

        # slike koje spolja (npr. iz test_teleoperation.py glavne petlje)
        # neko zeli da prikaze preko cv2.imshow -- namerno NE zovemo cv2
        # direktno iz spoljnog thread-a (OpenCV GUI nije pouzdano thread-safe,
        # pogotovo na Windows-u), nego samo upisujemo sliku ovde, a NASA
        # kamera nit (_camera_loop) je stvarno prikazuje. Kljuc = ime prozora.
        self.extra_frames = {}
        self._known_extra_windows = set()

        # trenutna (svaki frejm azurirana) prava 3D pozicija zgloba sake u
        # KAMERA frame-u (metri) -- dobijena RealSense deprojekcijom
        # (piksel + izmerena dubina -> 3D tacka), NE MediaPipe-ova naucena
        # procena. RealSense konvencija: X desno, Y dole, Z napred (dubina,
        # dalje od kamere) -- gledano IZ kamere KA sceni.
        self.current_wrist_3d = None

        # referentna (neutralna) pozicija, postavljena u trenutku SPACE pritiska
        self.reference_wrist_3d = None

        # "otklon dzojstika" -- (3,) u KAMERA frame-u, sada SVE tri
        # komponente u istim (pravim, metarskim) jedinicama -- deprojekcija
        # daje pravu dubinu, vise nema mesanja slika-frakcija + world-Z
        self.stick = np.zeros(3)


        self._enabled = False

        self._running = True



        # -------------------------
        # Robosuite states
        # -------------------------

        self._reset_internal_state()

        self._reset_state = 0



        # -------------------------
        # Keyboard
        # -------------------------

        self.listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )

        self.listener.start()



        # -------------------------
        # Camera thread
        # -------------------------

        self.thread = threading.Thread(
            target=self._camera_loop,
            daemon=True
        )

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



    def update_extra_frame(self, name, image_bgr):
        """
        Bezbedno (thread-safe) predaj sliku da je NASA kamera nit prikaze
        pod cv2 prozorom @name. Pozovi iz BILO KOG drugog thread-a (npr.
        glavne petlje u test_teleoperation.py) -- NIKAD ne zovi cv2.imshow
        direktno iz tog drugog thread-a, to je izvor problema koji smo
        upravo resile (prozori se tiho ne otvaraju).
        """
        with self.lock:
            self.extra_frames[name] = image_bgr



    @staticmethod
    def _deadzone(value, dz):
        """Mrtva zona: vrednosti unutar [-dz, dz] postaju 0, van te zone se
        NEPREKIDNO (bez skoka) nastavlja od 0 -- oduzima dz umesto da samo
        odseca, da ne bi bilo naglog skoka na granici zone."""
        if abs(value) < dz:
            return 0.0
        return value - dz * np.sign(value)



    # --------------------------------------------------
    # keyboard
    # --------------------------------------------------

    def _on_press(self,key):

        if key == Key.space:

            with self.lock:

                if not self._clutch_active:

                    self._clutch_active = True

                    if self.current_wrist_3d is not None:

                        self.reference_wrist_3d = (
                            self.current_wrist_3d.copy()
                        )

                        self.stick = np.zeros(3)  # cist pocetak svake nove sesije



    def _on_release(self,key):

        if key == Key.space:

            with self.lock:

                self._clutch_active = False



    # --------------------------------------------------
    # controller interface (legacy -- za robosuite input2action put koji
    # ionako NE koristimo u glavnom pipeline-u; ostavljeno da ne pukne ako
    # ga neko pozove, ne racunaj na ovo za rate-control logiku)
    # --------------------------------------------------

    def get_controller_state(self):

        t0 = time.perf_counter()


        with self.lock:

            current_stick = self.stick.copy()


        dstick = current_stick - self.last_stick

        self.last_stick = current_stick



        raw_drotation = (
            self.raw_drotation -
            self.last_drotation
        )


        self.last_drotation = (
            self.raw_drotation.copy()
        )


        t1 = time.perf_counter()


        self.control_times.append(
            t1 - t0
        )


        if time.perf_counter() - self.last_control_print > 1.0:


            if len(self.control_times):

                controller_hz = (
                    1 /
                    np.mean(self.control_times)
                )

                print(
                    f"Controller frequency: {controller_hz:.1f} Hz"
                )


            self.last_control_print = time.perf_counter()



        return dict(

            dpos=dstick,

            rotation=self.rotation,

            raw_drotation=raw_drotation,

            grasp=int(self.grasp),

            reset=self._reset_state,

            base_mode=int(self.base_mode),

        )



    # --------------------------------------------------
    # camera thread
    # --------------------------------------------------

    def _camera_loop(self):

        while self._running:


            loop_start = time.perf_counter()


            # -------------------------
            # camera read timing
            # -------------------------

            rs_frames = self.rs_pipeline.wait_for_frames()
            aligned = self.rs_align.process(rs_frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()


            camera_end = time.perf_counter()


            if not depth_frame or not color_frame:
                continue



            self.camera_times.append(
                camera_end-loop_start
            )



            frame = np.asanyarray(color_frame.get_data())

            # VAZNO: NE flipujemo ovde -- MediaPipe detekcija i deprojekcija
            # (rs2_deproject_pixel_to_point) moraju raditi na SIROVOM pikselu
            # koji odgovara stvarnim intrinsics-ima senzora. Flip radimo tek
            # na kraju, SAMO za prikaz (cv2.imshow), na kopiji.



            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            self.frame_timestamp_ms += 33



            # -------------------------
            # MediaPipe timing
            # -------------------------

            mp_start = time.perf_counter()


            result = self.detector.detect_for_video(
                mp_image,
                self.frame_timestamp_ms
            )


            mp_end = time.perf_counter()


            self.mediapipe_times.append(
                mp_end-mp_start
            )



            # -------------------------
            # Hand processing
            # -------------------------

            if result.hand_world_landmarks and result.hand_landmarks:


                world_landmarks = result.hand_world_landmarks[0]
                image_landmarks = result.hand_landmarks[0]


                wrist_world = world_landmarks[0]
                wrist_image = image_landmarks[0]

                # -------------------------
                # GRASP DETECTION (nepromenjeno -- world landmarks, pinch distanca)
                # -------------------------

                thumb_tip = world_landmarks[4]
                index_tip = world_landmarks[8]

                distance = np.sqrt(
                    (thumb_tip.x - index_tip.x)**2 +
                    (thumb_tip.y - index_tip.y)**2 +
                    (thumb_tip.z - index_tip.z)**2
                )

                is_grasping = distance < self.GRASP_THRESHOLD


                with self.lock:
                    self.grasp_states[
                        self.active_robot
                    ][self.active_arm_index] = is_grasping


                wrist_image_pos = np.array(
                    [wrist_image.x, wrist_image.y]
                )

                # -------------------------
                # DEPROJEKCIJA: piksel + prava dubina -> prava 3D tacka
                # -------------------------

                h, w = frame.shape[:2]
                px = int(np.clip(wrist_image_pos[0] * w, 0, w - 1))
                py = int(np.clip(wrist_image_pos[1] * h, 0, h - 1))

                depth_m = depth_frame.get_distance(px, py)  # vec u metrima

                # depth_m == 0 znaci "nema validnog ocitavanja" na tom pikselu
                # (van dometa senzora, IR refleksija, senka) -- preskoci ovaj
                # frejm umesto da koristis lazno nula-rastojanje
                if depth_m <= 0.0:
                    continue

                point_3d = rs.rs2_deproject_pixel_to_point(
                    self.rs_intrinsics, [px, py], depth_m
                )
                wrist_3d = np.array(point_3d)  # (X, Y, Z) metri, kamera frame


                with self.lock:

                    self.current_wrist_3d = wrist_3d.copy()


                    if (
                        self._enabled
                        and self._clutch_active
                        and self.reference_wrist_3d is not None
                    ):

                        offset_x = wrist_3d[0] - self.reference_wrist_3d[0]
                        offset_y = wrist_3d[1] - self.reference_wrist_3d[1]
                        offset_z = wrist_3d[2] - self.reference_wrist_3d[2]

                        offset_x = self._deadzone(offset_x, self.deadzone_xy)
                        offset_y = self._deadzone(offset_y, self.deadzone_xy)
                        offset_z = self._deadzone(offset_z, self.deadzone_z)

                        raw_stick = np.array([offset_x, offset_y, offset_z])

                        self.stick = (
                            self.stick_alpha * raw_stick
                            + (1 - self.stick_alpha) * self.stick
                        )



            # -------------------------
            # performance print
            # -------------------------

            if time.perf_counter() - self.last_perf_print > 1.0:


                camera_fps = (
                    1 /
                    np.mean(self.camera_times)
                )


                mp_latency = (
                    np.mean(self.mediapipe_times)
                    *1000
                )


                print("\n----- PERFORMANCE -----")

                print(
                    f"Camera FPS: {camera_fps:.1f}"
                )

                print(
                    f"MediaPipe latency: {mp_latency:.1f} ms"
                )

                print("-----------------------\n")


                self.last_perf_print = time.perf_counter()



            # -------------------------
            # display -- flip PRVI, pa TEK ONDA iscrtavanje na flipovanoj
            # slici (inace se i tekst flipuje pa se cita unazad kao u ogledalu)
            # -------------------------

            with self.lock:

                clutch = self._clutch_active
                stick_display = self.stick.copy()


            display_frame = cv2.flip(frame, 1)
            h, w = display_frame.shape[:2]


            text = (
                "TRACKING"
                if clutch
                else
                "PAUSED"
            )


            color = (
                (0,255,0)
                if clutch
                else
                (0,0,255)
            )


            cv2.putText(
                display_frame,
                text,
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )


            if clutch:
                cv2.putText(
                    display_frame,
                    f"stick: {stick_display.round(3)}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )



            if result.hand_landmarks:

                for lm in result.hand_landmarks[0]:

                    # zrcali X (w - x) da se tackica poklopi sa flipovanom slikom
                    x = w - 1 - int(lm.x * w)

                    y = int(lm.y * h)


                    cv2.circle(
                        display_frame,
                        (x,y),
                        3,
                        (0,255,0),
                        -1
                    )



            cv2.imshow(
                "MediaPipe Teleoperation",
                display_frame
            )

            # -- prikazi sve "extra" slike koje je neko drugi thread predao
            # preko update_extra_frame() -- SVE cv2.imshow/waitKey pozivi
            # ostaju u OVOJ jednoj niti, namerno, da se izbegne poznati
            # OpenCV multi-thread GUI problem (prozori se tiho ne otvaraju)
            with self.lock:
                extra_frames_copy = dict(self.extra_frames)

            for window_name, img in extra_frames_copy.items():
                if window_name not in self._known_extra_windows:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 400, 400)
                    self._known_extra_windows.add(window_name)
                cv2.imshow(window_name, img)


            cv2.waitKey(1)




    def close(self):

        self._running=False


        self.thread.join(timeout=2)


        self.rs_pipeline.stop()


        cv2.destroyAllWindows()


        self.listener.stop()