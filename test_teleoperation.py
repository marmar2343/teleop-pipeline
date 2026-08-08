"""
test_teleoperation.py

v4 -- RATE CONTROL + prava RGB-D dubina (RealSense D405), zamenjuje v3
(2.5D: slika X/Y + MediaPipe-ova NAUCENA world-landmark Z procena).

STA SE PROMENILO I ZASTO:

1) device.stick (iz mediapipe_device.py v4) je i dalje "otklon dzojstika"
   od referentne tacke, ali SADA sve tri komponente dolaze iz PRAVE
   dubinske deprojekcije (RealSense), ne mesavine slika-frakcija +
   naucena-Z kao pre. Sve tri ose su sada u ISTIM (pravim, metarskim)
   jedinicama.

2) target += velocity * dt (rate control) OSTAJE nepromenjeno -- i dalje
   vredno zadrzati zbog glatkoce/bezbednosti, samo sto ulazni signal sad
   dolazi sa mnogo manje sistematske greske nego pre.

3) Rotaciona kalibracija (calibrate_camera_to_robot_rotation iz
   calibration.py) i dalje NIJE pozvana ovde -- mapiranje ose je i dalje
   EKSPLICITNO postavljeno preko SIGN_* konstanti ispod. Ako se ikad vratis
   na fitovanu rotaciju, sad bi radila na mnogo cistijim podacima (prava
   dubina, ne monokularna procena) nego kad smo je prvi put probale.

PRVI TEST POSLE OVE IZMENE: pokreni, drzi SPACE, pomeri ruku SAMO ulevo-
udesno i proveri da li robot ide u ocekivanom smeru. Ako ne, okreni
SIGN_LATERAL. Ponovi za gore-dole (SIGN_VERTICAL) i napred-nazad (SIGN_FORWARD).
"""

import time

import cv2
import numpy as np
import robosuite as suite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"  # ispravna (ne-naopacka) orijentacija slike -- MORA pre suite.make()

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

from mediapipe_device import MediaPipeDevice
from ik_solver import DLS_IK_Solver
import hanoi_three_env  # noqa: F401 -- import registruje "HanoiThree" u robosuite (EnvMeta metaklasa)


# ============================================================
# MAPIRANJE OSA -- jedino mesto koje dirati kad menjas smerove
# ============================================================
# device.stick je SADA prava 3D tacka (RealSense deprojekcija), u kamera
# frame-u: X desno, Y dole, Z napred (dubina, dalje od kamere) -- gledano
# IZ kamere KA sceni (Intel-ova standardna konvencija).

SIGN_FORWARD = -1.0     # robot X (napred/nazad) <- stick[2] (RealSense dubina)
SIGN_LATERAL = -1.0    # robot Y (levo/desno)   <- stick[0] (RealSense X) -- OKRENUTO jer detekcija sad radi na sirovom (neflipovanom) frejmu, kamera je "ogledalo" (gleda te licem u lice)
SIGN_VERTICAL = -1.0   # robot Z (gore/dole)    <- stick[1] (RealSense Y, INVERTOVANO jer je dole=pozitivno u kamera frame-u)

# K_XY i K_Z su SADA uporedivije po redu velicine (obe ose su prave metre) --
# i dalje odvojene konstante jer stereo dubina i lateralna preciznost mogu
# imati razlicit "osecaj" pri koriscenju, pa vredi moci nezavisno podesiti
K_XY = 2.0   # m/s po metru lateralnog/vertikalnog otklona -- TUNABLE
K_Z = 3.0    # m/s po metru otklona dubine -- TUNABLE

MAX_LINEAR_SPEED = 0.4  # m/s, sigurnosno ogranicenje ukupne brzine hvataljke

# -- NELINEARNO SKALIRANJE (predlog mentora) --
# Ideja: mali otkloni (fina, precizna kontrola) treba da daju JOS manju
# brzinu nego linearno, a veliki otkloni (namerno brzo kretanje) treba da
# ostanu blizu pune brzine. STICK_MAX_* su "ocekivani maksimalni" otkloni
# posle deadzone-a, samo za normalizaciju krive -- ne moraju biti savrseno
# tacni, sluze da kriva bude u razumnom opsegu.
CURVE_EXPONENT = 2.0   # 1.0 = linearno (bez efekta), 2.0 = kvadratno, veci broj = izrazenija razlika fino/brzo
STICK_MAX_XY = 0.15    # ocekivan maksimalan lateralni/vertikalni otklon (m) posle deadzone-a
STICK_MAX_Z = 0.15     # ocekivan maksimalan otklon dubine (m) posle deadzone-a -- sad uporediv sa XY, jer je i Z prava metarska mera


def _apply_curve(value, max_expected, exponent):
    """
    Nelinearna kriva odziva -- cuva znak, normalizuje na [0,1] u odnosu na
    max_expected, stepenuje, pa vraca u originalnu skalu. Sa exponent=2:
    otklon od 50% max-a daje SAMO 25% izlaza (fina kontrola blizu centra),
    dok pun otklon i dalje daje pun izlaz (brzo kretanje kad namerno
    odmakenes ruku daleko od reference).
    """
    normalized = np.clip(abs(value) / max_expected, 0.0, 1.0)
    shaped = normalized ** exponent
    return np.sign(value) * shaped * max_expected


def build_env(control_freq=30, use_cameras=True):
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
        color_code_pegs=True,  # narandzasto=izvor, zeleno=cilj -- SAMO za tvoje testiranje, iskljuci za snimanje pravih demonstracija
        has_renderer=True,             # zivi prikaz (env.render()) -- OBAVEZNO True za ovu petlju
        has_offscreen_renderer=use_cameras,  # OBAVEZNO True ako koristis use_camera_obs -- ali ne iskljucuje has_renderer, mogu oba istovremeno
        use_camera_obs=use_cameras,
        camera_names=["sideview", "robot0_eye_in_hand"] if use_cameras else None,
        camera_heights=256 if use_cameras else None,
        camera_widths=256 if use_cameras else None,
        control_freq=control_freq,
        horizon=2000,            # Hanoj je slozeniji zadatak od Lift-a, daj vise vremena
        ignore_done=True,
    )
    return VisualizationWrapper(env)


def stick_to_velocity(stick):
    """Otklon dzojstika (mesovite jedinice) -> zeljena brzina hvataljke (m/s, robot frame)."""
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


if __name__ == "__main__":
    env = build_env()
    obs = env.reset()
    env.render()
    env.robots[0].print_action_info()

    solver = DLS_IK_Solver(env, arm="right", damping=0.05, step_size=0.5, max_joint_step=0.2)

    device = MediaPipeDevice(env=env, model_path="hand_landmarker.task")
    device.start_control()

    dt = 1.0 / env.control_freq

    target_ee_pos = solver.get_eef_position()
    prev_clutch = False
    step_count = 0

    print("Pokrenuto (rate control). Drzi SPACE da pratis rukom. Ctrl+C u terminalu da prekines.")

    try:
        while True:
            with device.lock:
                clutch_now = device._clutch_active
                stick = device.stick.copy()

            q_current = solver.get_current_qpos()

            if clutch_now and not prev_clutch:
                # clutch upravo aktiviran -- re-sinhronizuj target sa STVARNOM
                # pozicijom hvataljke, da se ne akumulira razmak izmedju
                # komandovanog i stvarnog stanja iz prethodnih sesija
                target_ee_pos = solver.get_eef_position()

            if clutch_now:
                velocity = stick_to_velocity(stick)
                target_ee_pos = target_ee_pos + velocity * dt
                q_target = solver.step(q_current, target_ee_pos)
            else:
                # clutch nije aktivan -- drzi robota na trenutnoj pozi,
                # NE integrisi (target_ee_pos ostaje zamrznut do sledeceg engage-a)
                q_target = q_current

            prev_clutch = clutch_now

            gripper_action = np.array([1.0 if device.grasp else -1.0])
            action = np.concatenate([q_target, gripper_action])

            assert len(action) == env.action_dim, (
                f"Ocekivano {env.action_dim} dim akcije, dobijeno {len(action)} -- "
                f"proveri broj zglobova ruke / dimenziju grippera"
            )

            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(dt)

            # -- predaj oba kamera strima device-u da ih ON prikaze (u SVOM
            # thread-u) -- NIKAD ne zovi cv2.imshow direktno ovde, to je
            # bio uzrok da se prozori tiho ne otvaraju (dva thread-a rade
            # cv2 GUI istovremeno)
            if "sideview_image" in obs:
                device.update_extra_frame("sideview", cv2.cvtColor(obs["sideview_image"], cv2.COLOR_RGB2BGR))
            if "robot0_eye_in_hand_image" in obs:
                device.update_extra_frame("robot0_eye_in_hand", cv2.cvtColor(obs["robot0_eye_in_hand_image"], cv2.COLOR_RGB2BGR))

            step_count += 1
            if step_count % 10 == 0:
                success = env.env._check_success()  # env.env: HanoiThree ispod VisualizationWrapper-a
                status = "RESENO! 🎉" if success else "u toku"
                print(f"clutch={clutch_now}, stick={stick.round(3)}, ee={solver.get_eef_position().round(3)}, status={status}")

            if done:
                print("Epizoda zavrsena, resetujem...")
                obs = env.reset()
                target_ee_pos = solver.get_eef_position()

    except KeyboardInterrupt:
        print("Prekinuto.")
    finally:
        device.close()
        env.close()