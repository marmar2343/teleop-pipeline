"""
test_teleoperation.py

v3 -- RATE CONTROL + 2.5D, zamenjuje raniji position-control pristup koji je
zahtevao punu 3x3 rotacionu kalibraciju (calibration.py) i bio osetljiv na
MediaPipe-ovu monokularnu gresku skale.

STA SE PROMENILO I ZASTO:

1) device.stick (iz mediapipe_device.py v3) je "otklon dzojstika" od
   referentne tacke, NE apsolutna pozicija. X/Y dolaze sa SLIKE (pouzdanije),
   Z i dalje iz world landmarks (manje pouzdano, ali sad manje bitno -- videti nize).

2) Umesto target = referenca + hand_delta (position control), sada:
       target += velocity * dt   (integrise se SVAKI korak, rate control)
   Greska u proceni skale sad utice na BRZINU kretanja za dati otklon ruke,
   ne na finalnu poziciju hvataljke -- mnogo tolerantnije na netacnu
   MediaPipe procenu.

3) Rotaciona kalibracija (calibrate_camera_to_robot_rotation) VISE NIJE
   POTREBNA -- mapiranje ose je sada EKSPLICITNO postavljeno ispod
   (SIGN_* konstante), umesto fitovano iz 3 nesigurna pokreta. Mnogo
   robusnije, samo treba proveriti/okrenuti znake ako se robot krece u
   pogresnom smeru (uputstvo ispod). calibrate_scale takodje nije potrebna
   iz istog razloga -- K_XY/K_Z su obicni gain-ovi za podesavanje, ne
   merenje "prave" fizicke skale.

PRVI TEST POSLE OVE IZMENE: pokreni, drzi SPACE, pomeri ruku SAMO ulevo-
udesno i proveri da li robot ide u ocekivanom smeru. Ako ne, okreni
SIGN_LATERAL. Ponovi za gore-dole (SIGN_VERTICAL) i napred-nazad (SIGN_FORWARD).
"""

import time

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

from mediapipe_device import MediaPipeDevice
from ik_solver import DLS_IK_Solver
import hanoi_three_env  


# ============================================================
# MAPIRANJE OSA -- jedino mesto koje dirati kad menjas smerove
# ============================================================
# device.stick[0] = otklon na SLICI, levo-desno (X na slici)
# device.stick[1] = otklon na SLICI, gore-dole (Y na slici, dole = pozitivno!)
# device.stick[2] = otklon dubine iz world landmarks (Z, metri)

SIGN_FORWARD = 1.0     # robot X (napred/nazad) <- stick[2] (dubina)
SIGN_LATERAL = 1.0     # robot Y (levo/desno)   <- stick[0] (slika X)
SIGN_VERTICAL = -1.0   # robot Z (gore/dole)    <- stick[1] (slika Y, INVERTOVANO jer dole=pozitivno na slici)

K_XY = 0.6   # m/s po jedinici normalizovanog otklona na slici (0-1 opseg) -- TUNABLE
K_Z = 2.5    # m/s po metru otklona dubine -- TUNABLE, verovatno treba podesiti empirijski

MAX_LINEAR_SPEED = 0.15  # m/s, sigurnosno ogranicenje ukupne brzine hvataljke

# nelinearno skaliranje
CURVE_EXPONENT = 2.0   # 1.0 = linearno (bez efekta), 2.0 = kvadratno, veci broj = izrazenija razlika fino/brzo
STICK_MAX_XY = 0.25    # ocekivan maksimalan otklon na slici posle deadzone-a
STICK_MAX_Z = 0.12     # ocekivan maksimalan otklon dubine (m) posle deadzone-a


def _apply_curve(value, max_expected, exponent):
    
    normalized = np.clip(abs(value) / max_expected, 0.0, 1.0)
    shaped = normalized ** exponent
    return np.sign(value) * shaped * max_expected


def build_env(control_freq=30):
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
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,   # HanoiThree podrazumevano trazi True -- ovde nam ne treba jos (Faza 4)
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