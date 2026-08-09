"""
mediapipe_device.py

MediaPipe hand teleoperation device za robosuite -- v4, RATE CONTROL + RealSense D405.

PROMENA U ODNOSU NA v3 (2.5D: slika X/Y + MediaPipe world-landmark Z):

  - X/Y/Z pozicije zgloba I SADA merenje pinch-a (grasp) idu preko PRAVE
    dubinske deprojekcije -- hand_world_landmarks (MediaPipe naucena
    metricka procena) se VISE NIGDE ne koristi za merenje. Razlog izmene
    pinch-a: hand_world_landmarks ima priblizno fiksnu apsolutnu gresku po
    vrhu prsta -- zanemarljivo za veliko rastojanje otvorene sake, ali
    DOMINANTNO kod sitnog pinch rastojanja, otud vrednosti koje su jako
    varirale sa udaljenoscu od kamere dok je stara verzija to koristila.
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
        grasp_close_threshold=0.035,  # ISPOD ovoga: pusti->zatvori (stroziji prag)
        grasp_open_threshold=0.05,    # IZNAD ovoga: zatvoreno->otvori (blazi prag) -- razlika izmedju ova dva je HISTEREZA
        grasp_alpha=0.4,               # EMA glacanje NAD sirovim pinch rastojanjem, PRE odluke
    ):

        super().__init__(env)

        self.GRASP_CLOSE_THRESHOLD = grasp_close_threshold
        self.GRASP_OPEN_THRESHOLD = grasp_open_threshold
        self.grasp_alpha = grasp_alpha
        self._pinch_distance_smooth = grasp_open_threshold  # pocni kao "otvoreno"

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
        self._camera_start_time = time.perf_counter()



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
        self.status_lines = []

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



    def update_status(self, lines):
        """
        Bezbedno (thread-safe) predaj listu stringova da se prikazu na
        statusnom panelu u dashboard-u (npr. pozicija hvataljke, status
        zadatka -- stvari koje samo glavna petlja zna). Isti princip kao
        update_extra_frame -- ne crta se ovde, samo se cuva za _camera_loop.
        """
        with self.lock:
            self.status_lines = list(lines)



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


            # STVARNO proteklo vreme, ne fiksnih +33ms -- ako isporuka
            # frejmova ikad zastane/ubrza, MediaPipe-ov VIDEO mod (koji
            # interno koristi timestamp za sopstveno vremensko pracenje)
            # sad dobija tacnu informaciju, ne pretpostavku
            self.frame_timestamp_ms = int((time.perf_counter() - self._camera_start_time) * 1000)



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


                # NAPOMENA: hand_world_landmarks se vise NE koristi za
                # merenje (ni pinch ni pozicija) -- samo se i dalje trazi u
                # detect_for_video pozivu iznad, uslov ispod ostaje kao
                # razuman "da li je saka pouzdano detektovana" proveru.
                image_landmarks = result.hand_landmarks[0]
                wrist_image = image_landmarks[0]

                h, w = frame.shape[:2]

                # -------------------------
                # GRASP DETECTION -- SADA preko prave RealSense dubine na
                # oba vrha prsta (isti postupak kao za zglob sake ispod),
                # umesto MediaPipe-ove naucene hand_world_landmarks procene.
                #
                # Razlog izmene: hand_world_landmarks ima priblizno FIKSNU
                # apsolutnu gresku po vrhu prsta (par mm) -- zanemarljivo za
                # veliko rastojanje otvorene sake (~8cm), ali DOMINANTNO kad
                # se meri sitno rastojanje tokom pinch-a (par mm), otud
                # vrednosti koje jako variraju sa udaljenoscu od kamere.
                # Prava dubina nema tu manu, ista tehnika koja vec pouzdano
                # radi za poziciju zgloba.
                #
                # EMA izgladjivanje + histereza (dva praga) ostaju
                # nepromenjeni ispod -- i dalje potrebni protiv sitnog suma,
                # samo sad na mnogo pouzdanijem sirovom signalu.
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
                    thumb_3d = np.array(rs.rs2_deproject_pixel_to_point(
                        self.rs_intrinsics, [thumb_px, thumb_py], thumb_depth_m
                    ))
                    index_3d = np.array(rs.rs2_deproject_pixel_to_point(
                        self.rs_intrinsics, [index_px, index_py], index_depth_m
                    ))
                    raw_distance = np.linalg.norm(thumb_3d - index_3d)
                else:
                    # nevalidna dubina na vrhu prsta (cesto bas TOKOM pinch-a,
                    # kad prsti delimicno zaklanjaju jedan drugog) -- zadrzi
                    # PRETHODNU izgladjenu vrednost umesto da racunas sa
                    # ocigledno pogresnim brojem za ovaj jedan frejm
                    raw_distance = self._pinch_distance_smooth

                with self.lock:
                    self._pinch_distance_smooth = (
                        self.grasp_alpha * raw_distance
                        + (1 - self.grasp_alpha) * self._pinch_distance_smooth
                    )
                    smoothed = self._pinch_distance_smooth

                    currently_grasping = self.grasp_states[self.active_robot][self.active_arm_index]

                    if currently_grasping:
                        # vec drzi -- pusti SAMO ako rastojanje ubedljivo
                        # preraste GORNJI (blazi) prag
                        is_grasping = smoothed < self.GRASP_OPEN_THRESHOLD
                    else:
                        # trenutno otvoreno -- zatvori SAMO ako rastojanje
                        # padne ispod DONJEG (strozeg) praga
                        is_grasping = smoothed < self.GRASP_CLOSE_THRESHOLD

                    self.grasp_states[
                        self.active_robot
                    ][self.active_arm_index] = is_grasping


                wrist_image_pos = np.array(
                    [wrist_image.x, wrist_image.y]
                )

                # -------------------------
                # DEPROJEKCIJA: piksel + prava dubina -> prava 3D tacka
                # -------------------------

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
            # display -- prava kamera (RealSense/MediaPipe) se NE prikazuje
            # (na tvoj zahtev) -- ali TRACKING status i pinch vrednost i
            # dalje moraju negde da se vide, pa idu u TEKST status panel
            # umesto da se ispisuju preko slike
            # -------------------------

            with self.lock:

                clutch = self._clutch_active
                stick_display = self.stick.copy()
                pinch_display = self._pinch_distance_smooth
                extra_frames_copy = dict(self.extra_frames)
                status_lines_copy = list(self.status_lines)


            device_status_lines = [
                f"Tracking: {'DA' if clutch else 'ne'}",
                f"Pinch (glatko): {pinch_display:.4f}",
                f"  zatvori<{self.GRASP_CLOSE_THRESHOLD:.3f}  otvori>{self.GRASP_OPEN_THRESHOLD:.3f}",
                f"Stick: {stick_display.round(3)}" if clutch else "Stick: (clutch nije aktivan)",
                "",
            ]

            # -- prava kamera, VRACENA u prikaz -- flip PRVI, tackice sa
            # zrcaljenim X (poklapaju se sa flipovanom slikom), BEZ teksta
            # preko slike (tekst je vec u status panelu, ne duplira se ovde)
            real_cam_frame = cv2.flip(frame, 1)
            if result.hand_landmarks:
                rh, rw = real_cam_frame.shape[:2]
                for lm in result.hand_landmarks[0]:
                    x = rw - 1 - int(lm.x * rw)
                    y = int(lm.y * rh)
                    cv2.circle(real_cam_frame, (x, y), 3, (0, 255, 0), -1)

            dashboard = self._build_dashboard(
                real_cam_frame, extra_frames_copy, device_status_lines + status_lines_copy
            )

            if "dashboard" not in self._known_extra_windows:
                cv2.namedWindow("Teleoperacija -- Dashboard", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Teleoperacija -- Dashboard", 900, 700)
                self._known_extra_windows.add("dashboard")

            cv2.imshow("Teleoperacija -- Dashboard", dashboard)


            cv2.waitKey(1)



    def _build_dashboard(self, real_camera_frame, extra_frames, status_lines, tile_size=320):
        """
        Slaze pravu kameru (RealSense, sa tackicama detekcije sake) +
        agentview + robot0_eye_in_hand (simulacione kamere) + tekstualni
        status panel u JEDNU sliku, raspored 2x2.

        sideview NAMERNO nije ovde (redundantan sa agentview za live
        pregled) -- i dalje se snima za dataset preko camera_names u
        test_teleoperation.py, samo se ne prikazuje uzivo.

        tile_size je NAMERNO kvadratan (ne pravougaon tile_w/tile_h kao
        ranije) da odgovara kvadratnoj rezoluciji simulacionih kamera --
        razvlacenje kvadratne slike u pravougaoni tile je verovatno bio
        glavni uzrok "uzasnog kvaliteta" (izobljena/razvucena slika).
        cv2.INTER_CUBIC umesto podrazumevanog INTER_LINEAR za blazi rezultat
        pri increasingu velicine.

        Napomena: RoboSuite-ov sopstveni ziv 3D prikaz (env.render()) NIJE
        ovde -- to je poseban nativni MuJoCo viewer, nema sirove piksele
        dostupne preko cv2, ne moze se ubaciti u ovaj kolaz.
        """
        def prep(img, title):
            if img is None:
                tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
            # traka sa naslovom preko vrha, radi citljivosti bez obzira na sadrzaj slike ispod
            cv2.rectangle(tile, (0, 0), (tile_size, 22), (40, 40, 40), -1)
            cv2.putText(tile, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            return tile

        real_tile = prep(real_camera_frame, "Kamera (ruka)")
        agent_tile = prep(extra_frames.get("agentview"), "Agentview kamera (sim)")
        eye_tile = prep(extra_frames.get("robot0_eye_in_hand"), "Wrist kamera (sim)")

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