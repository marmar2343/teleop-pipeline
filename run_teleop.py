"""
run_teleop.py

FAZA 4 -- snimanje pravih demonstracija za VLA dataset.

Ovo je pun spoj svega izgradjenog do sada:
    MediaPipeDevice (RealSense, rate control, 3D deprojekcija)
    -> DLS_IK_Solver (pozicija + zakljucana orijentacija)
    -> HanoiThree environment
    -> DataCollectionWrapper (RoboSuite-ov meh. za snimanje demonstracija)

Struktura petlje i gather_demonstrations_as_hdf5() funkcija su NAMERNO
bazirane na zvanicnom robosuite/scripts/collect_human_demonstrations.py
(kopirano/prilagodjeno odatle, ne izmisljeno) -- razlika je STO se umesto
device.input2action() (za OSC/IK_POSE kontrolere) koristi TVOJ rate-control
lanac (device.stick -> stick_to_velocity -> DLS_IK_Solver), pošto koristis
JOINT_POSITION kontroler sa sopstvenom IK.

VAZNA NAPOMENA o "uspesnosti" epizode:
DataCollectionWrapper AUTOMATSKI prati env._check_success() na svakom
koraku (videti izvorni kod -- self.env._check_success() se poziva u
step()) i cuva SAMO uspesne epizode kad se gather_demonstrations_as_hdf5
pozove. TO NE ukljucuje tvoju check_hanoi_legality() proveru (poteza
tokom cele epizode, ne samo finalnog stanja) -- ta provera se NAMERNO radi
KAO POSEBAN POST-PROCESSING KORAK, u zasebnom fajlu (validate_dataset.py),
NE ovde uzivo. Razlog: ne zelimo da "kaznjavamo"/prekidamo operatera usred
prirodnog resavanja zadatka -- snimi sve, filtriraj posle.
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

macros.IMAGE_CONVENTION = "opencv"  # ispravna (ne-naopacka) orijentacija slike -- MORA pre suite.make()

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from mediapipe_device import MediaPipeDevice
from ik_solver import DLS_IK_Solver
import hanoi_three_env  # noqa: F401 -- import registruje "HanoiThree" u robosuite (EnvMeta metaklasa)

# Iste konstante kao u test_teleoperation.py -- ako menjas SIGN_*/K_XY/K_Z
# tamo dok podesavas osecaj kontrole, prekopiraj i ovde (namerno odvojeno,
# ne deljeno kroz import, da slucajno neka eksperimentalna vrednost iz
# testiranja ne zavrsi u snimljenim demonstracijama bez da to primetis)
SIGN_FORWARD = 1.0
SIGN_LATERAL = -1.0
SIGN_VERTICAL = -1.0

K_XY = 2.0
K_Z = 2.0

MAX_LINEAR_SPEED = 0.3

CURVE_EXPONENT = 2.0
STICK_MAX_XY = 0.15
STICK_MAX_Z = 0.15

# koliko UZASTOPNIH koraka _check_success() mora biti True pre nego sto
# epizodu proglasimo zavrsenom -- sprecava da se epizoda "slucajno" zavrsi
# na jedan trenutni fizicki drhtaj/jitter oko granice uspeha (isti obrazac
# kao task_completion_hold_count u zvanicnom robosuite skriptu)
SUCCESS_HOLD_STEPS = 10


def _apply_curve(value, max_expected, exponent):
    normalized = np.clip(abs(value) / max_expected, 0.0, 1.0)
    shaped = normalized ** exponent
    return np.sign(value) * shaped * max_expected


def stick_to_velocity(stick):
    """Otklon dzojstika (prava 3D tacka, RealSense frame) -> zeljena brzina hvataljke (m/s, robot frame)."""
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
    Spaja .npz fajlove snimljene u @directory u JEDAN .hdf5 fajl.

    Preuzeto/prilagodjeno direktno iz
    robosuite/scripts/collect_human_demonstrations.py (proveren, zvanican
    izvor -- ne izmisljeno). Cuva SAMO epizode koje su bile uspesne
    (dic["successful"] -- automatski prati DataCollectionWrapper preko
    _check_success() na svakom koraku).

    NAPOMENA -- APPEND, ne prepisivanje: ako @out_dir/demo.hdf5 VEC postoji
    (npr. iz prethodne sesije), NOVE uspesne epizode se DODAJU, numerisanje
    (demo_N) nastavlja tamo gde je proslo pokretanje stalo -- ne pravi se
    nov fajl niti se stari brise. Da zapocnes NOV, prazan dataset, obrisi
    demo.hdf5 rucno.

    Struktura izlaznog hdf5:
        data (grupa)
            date, time, repository_version, env, env_info (atributi)
            demo_1 (grupa) -- svaka uspesna epizoda
                model_file (atribut) -- MJCF xml string
                states (dataset) -- flattened mujoco stanja
                actions (dataset) -- akcije poslate u toj epizodi
            demo_2, ...
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")

    file_existed = os.path.exists(hdf5_path)
    f = h5py.File(hdf5_path, "a")  # "a" = append -- pravi fajl ako ne postoji, otvara za dopisivanje ako postoji

    if "data" not in f:
        grp = f.create_group("data")
    else:
        grp = f["data"]

    # nastavi numerisanje od najveceg POSTOJECEG demo_N (0 ako je fajl nov)
    existing_nums = [int(k.split("_")[1]) for k in grp.keys() if k.startswith("demo_")]
    num_eps = max(existing_nums) if existing_nums else 0
    n_before = num_eps

    # KLJUCNO za append mod: gather() ponovo skenira CEO tmp_directory
    # svaki put (ne samo nove foldere), pa bez ovoga bi se VEC obradjene
    # epizode dodale PONOVO kao duplikat svaki put kad se gather() pozove
    # posle sledece demonstracije unutar ISTE sesije. Cuvamo listu vec
    # obradjenih imena foldera kao atribut (perzistentno i kroz sesije).
    processed = set(json.loads(grp.attrs.get("processed_episodes", "[]")))

    env_name = grp.attrs.get("env", None)

    for ep_directory in sorted(os.listdir(directory)):  # sortirano -- ime foldera sadrzi timestamp, pa je ovo hronoloski red
        if ep_directory in processed:
            continue  # vec obradjeno u ranijem pozivu gather()-a, preskoci da ne dupliras

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

        if len(states) == 0:
            continue  # jos NIJE flush-ovano na disk -- NE oznacavaj kao obradjeno, probaj ponovo sledeci put

        processed.add(ep_directory)  # tek OVDE, kad znamo da folder stvarno ima sadrzaj

        if success:
            print(f"[gather] {ep_directory}: USPESNA, cuvam u dataset")
            # poslednje stanje se brise -- DataCollectionWrapper snima
            # stanje POSLE izvrsene akcije, pa ostane jedno visak stanje
            # na kraju (isti razlog naveden u zvanicnom skriptu)
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as xml_f:
                xml_str = xml_f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print(f"[gather] {ep_directory}: NEUSPESNA, preskacem")

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
    print(f"[gather] {verb} fajl -- {added} novih ovog puta, ukupno {num_eps} uspesnih epizoda u {hdf5_path}")
    return hdf5_path


def build_env(control_freq=30, live_camera_names=("agentview",)):
    """
    live_camera_names -- koje kamere se renderuju UZIVO (za dashboard) tokom
    snimanja. Podrazumevano SAMO ("agentview",) -- merenjem potvrdjeno da je
    svaka dodatna kamera realan trosak (jedna kamera 320x320 je izmerena na
    ~135ms/korak u ovom -- verovatno sporijem od tvog -- sandboxu, tri kamere
    na ~423ms/korak). Prosledi () ili None da ih potpuno iskljucis (najbrze,
    kao pre), ili dodaj "robot0_eye_in_hand"/"sideview" ako ti treba precizniji
    uvid i tvoj hardver to podnosi.

    NAPOMENA: za DataCollectionWrapper slike NISU potrebne uzivo (generisu se
    naknadno replay-ovanjem) -- ovo je cisto radi TVOG pracenja dok snimas.
    """
    controller_config = load_composite_controller_config(controller="BASIC")
    controller_config["body_parts"]["right"]["type"] = "JOINT_POSITION"
    controller_config["body_parts"]["right"]["input_type"] = "absolute"
    controller_config["body_parts"]["right"]["interpolation"] = "linear"

    config = {
        "env_name": "HanoiThree",
        "robots": "Panda",
        "controller_configs": controller_config,
    }

    live_camera_names = list(live_camera_names) if live_camera_names else None
    use_cameras = bool(live_camera_names)

    env = suite.make(
        **config,
        source_peg_idx=0,
        target_peg_idx=2,
        randomize_pegs=True,   # nasumican izvor/cilj na svaki reset -- vise raznovrsnosti u datasetu
        color_code_pegs=True,  # narandzasto=izvor, zeleno=cilj -- sad ispravno prati randomize_pegs (dinamicki, ne samo pri pravljenju scene)
        has_renderer=True,
        hard_reset=False,  # KLJUCNO -- bez ovoga, svaki env.reset() pravi NOV env.sim
                           # objekat, i solver (konstruisan JEDNOM, pre petlje demonstracija)
                           # ostaje "zaglavljen" gledajuci u stару, vise-ne-azuriranu simulaciju.
                           # Struktura scene (kocke, klinovi) se ionako ne menja izmedju
                           # epizoda -- samo pozicije, sto _reset_internal() vec ispravno radi
                           # bez potrebe za punim rebuild-om modela.
        has_offscreen_renderer=use_cameras,
        use_camera_obs=use_cameras,
        camera_names=live_camera_names,
        camera_heights=320 if use_cameras else None,
        camera_widths=320 if use_cameras else None,
        control_freq=control_freq,
        horizon=9000,  # ~5 minuta na 30Hz -- pravo resavanje Hanoja (7 poteza,
                       # svaki sa hvatanjem/pomeranjem/spustanjem, plus vreme za
                       # ispravljanje gresaka kao sto je ispala kocka) realno
                       # traje mnogo duze od pocetnih 2000 koraka (~66s)
        ignore_done=True,
    )

    env_info = json.dumps(config)

    env = VisualizationWrapper(env)

    return env, env_info


def collect_one_demonstration(env, device, ik_kwargs, dt):
    """
    Snima JEDNU demonstraciju -- petlja traje dok operater ne resi zadatak
    (_check_success() drzi SUCCESS_HOLD_STEPS uzastopnih koraka) ili dok se
    ne istekne horizon. env MORA vec biti omotan DataCollectionWrapper-om
    (ovaj poziv koristi env.reset()/env.step() koje taj wrapper prati).

    NAPOMENA: solver se NAMERNO pravi OVDE, SVEZE, posle svakog env.reset()
    poziva -- ne prosledjuje se gotov kao ranije. Razlog: DataCollectionWrapper
    ima SOPSTVEN mehanizam (reset_from_xml_string) koji moze da zameni
    env.sim objekat NEZAVISNO od hard_reset postavke na samom environment-u
    -- solver napravljen PRE tog reseta bi ostao da gleda u zastarelu
    referencu (isti koren problema kao originalni hard_reset bag, samo
    izazvan od strane wrapper-a, ne HanoiThree-a). DLS_IK_Solver.__init__
    je jeftin poziv (samo keshira indekse), pa nema stvarne cene da se
    pravi iznova svaku epizodu.

    Returns:
        legal (bool): da li check_hanoi_legality() NIKAD nije prijavio
            prekrsaj tokom cele epizode -- za tvoju informaciju/log, NE
            utice na DataCollectionWrapper-ovu sopstvenu (samo
            _check_success() bazirana) odluku o cuvanju.
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

    print(f"\n=== Nova demonstracija -- izvor={hanoi_env.source_peg_idx}, cilj={hanoi_env.target_peg_idx} ===")
    print("Drzi SPACE da pratis rukom. Resi zadatak da automatski zavrsis ovu demonstraciju.")

    while True:
        with device.lock:
            clutch_now = device._clutch_active
            stick = device.stick.copy()

        q_current = solver.get_current_qpos()

        if clutch_now and not prev_clutch:
            target_ee_pos = solver.get_eef_position()

        if clutch_now:
            velocity = stick_to_velocity(stick)
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

        # -- kamere i status na dashboard, isto kao u test_teleoperation.py --
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
            f"[SNIMANJE] korak {step_count}",
            f"Clutch: {'DA' if clutch_now else 'ne'}",
            f"Grasp: {'DA' if device.grasp else 'ne'}",
            f"EE: {solver.get_eef_position().round(3)}",
            "",
            "RESENO! :)" if success else "u toku...",
            "Legalno: DA" if legal else f"NELEGALNO: {illegal_reason}",
            f"(hold: {success_hold_count}/{SUCCESS_HOLD_STEPS})",
            f"Preostalo vreme: {(env.horizon - step_count) * dt:.0f}s",
        ])

        if success:
            success_hold_count += 1
        else:
            success_hold_count = 0

        if success_hold_count >= SUCCESS_HOLD_STEPS:
            print(f"Demonstracija zavrsena (uspesno drzano {SUCCESS_HOLD_STEPS} koraka).")
            break

        steps_remaining = env.horizon - step_count
        seconds_remaining = steps_remaining * dt

        # upozorenje na 30s i 10s pre isteka -- da ne bude iznenadjenje
        if steps_remaining in (int(30 / dt), int(10 / dt)):
            print(f"UPOZORENJE: jos ~{seconds_remaining:.0f}s pre nego sto ova demonstracija istekne bez uspeha.")

        if step_count >= env.horizon:
            print("Isteklo vreme (horizon) bez uspeha -- demonstracija se NECE sacuvati (DataCollectionWrapper zahteva uspeh).")
            break

    if not episode_always_legal:
        print("UPOZORENJE: ova epizoda je bar jednom prekrsila Hanoj pravilo tokom izvodjenja "
              "(videces detalje u validate_dataset.py posle snimanja) -- DataCollectionWrapper "
              "je i dalje cuva ako je zavrsila uspesno, jer prati samo finalni ishod uzivo.")

    return episode_always_legal


if __name__ == "__main__":
    env, env_info = build_env()

    # NAPOMENA: solver se VISE NE pravi ovde -- pravi se sveze unutar
    # collect_one_demonstration(), posle SVAKOG reset()-a (videti napomenu
    # tamo za razlog). Ovde samo cuvamo parametre da se prosledjuju.
    ik_kwargs = dict(arm="right", damping=0.05, step_size=0.5, max_joint_step=0.2)
    device = MediaPipeDevice(env=env, model_path="hand_landmarker.task")

    dt = 1.0 / env.control_freq

    # -- omotaji za snimanje: DataCollectionWrapper SPOLJA, redosled preuzet
    # direktno iz zvanicnog robosuite skripta --
    # tmp_directory OSTAJE per-sesijski (privremeno, ciscenje nije bitno --
    # sadrzaj se svaki put SPOJI u trajni new_dir/demo.hdf5 pre nego sto
    # sesija zavrsi)
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # FIKSNA putanja -- ista kroz SVE sesije/pokretanja, ne novi timestamp
    # svaki put -- gather_demonstrations_as_hdf5() dopisuje (append), pa
    # demo_1, demo_2, demo_3... nastavljaju kroz vise dana/sesija u ISTI fajl
    new_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demonstracije", "hanoi_dataset")
    os.makedirs(new_dir, exist_ok=True)

    print(f"Sirovi snimci (privremeno): {tmp_directory}")
    print(f"Finalni .hdf5 (dopisuje se kroz sve sesije): {new_dir}/demo.hdf5")
    print("Ctrl+C u terminalu da prekines seriju snimanja u bilo kom trenutku.\n")

    try:
        while True:
            collect_one_demonstration(env, device, ik_kwargs, dt)

            # BITNO: DataCollectionWrapper ne pise na disk odmah -- cuva u
            # memoriji dok se ne pozove _flush() (sto se inace desava tek
            # na SLEDECI env.reset(), na svakih flush_freq=100 koraka, ili
            # na env.close()). Bez ovog eksplicitnog poziva, gather_demonstrations_as_hdf5()
            # bi citala prazan direktorijum za epizodu koja se TEK zavrsila
            # -- ovo je otkriveno stvarnim testom, ne pretpostavkom.
            env._flush()

            # gather se poziva POSLE SVAKE demonstracije (ne samo na kraju)
            # -- ako sesija negde crashuje, ne gubis sve prethodno snimljeno
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)

    except KeyboardInterrupt:
        print("\nPrekinuto -- poslednji .hdf5 je vec sacuvan (azuriran posle svake epizode).")
    finally:
        device.close()
        env.close()