"""
inspect_dataset.py

Brz alat za pregled sadrzaja demo.hdf5 fajla (formata koji pravi
run_teleop.py preko gather_demonstrations_as_hdf5()).

Upotreba:
    python inspect_dataset.py demonstracije/<timestamp>/demo.hdf5
    python inspect_dataset.py demonstracije/<timestamp>/demo.hdf5 --replay demo_1

Bez --replay, samo ispisuje strukturu i metapodatke (broj demonstracija,
duzina svake, verzija robosuite-a, itd.) -- ne treba mu robot/kamera,
samo h5py.

Sa --replay <ime_demoa>, VIZUELNO PUSTA tu demonstraciju u RoboSuite
vizuelizatoru -- ucitava sacuvani model.xml (tacna scena kakva je bila u
tom trenutku) i redom postavlja svako sacuvano stanje, isti princip koji
robomimic-ov playback_dataset.py koristi.
"""

import sys
import time

import h5py
import numpy as np


def inspect(path):
    with h5py.File(path, "r") as f:
        data = f["data"]

        print(f"Fajl: {path}")
        print(f"Environment: {data.attrs.get('env', '?')}")
        print(f"Datum: {data.attrs.get('date', '?')} {data.attrs.get('time', '?')}")
        print(f"Robosuite verzija: {data.attrs.get('repository_version', '?')}")
        print()

        demo_names = sorted(data.keys(), key=lambda n: int(n.split("_")[1]))
        print(f"Broj demonstracija: {len(demo_names)}")
        print()

        for name in demo_names:
            demo = data[name]
            n_steps = demo["states"].shape[0]
            print(
                f"  {name}: {n_steps} koraka  "
                f"(states shape={demo['states'].shape}, actions shape={demo['actions'].shape})"
            )

        if demo_names:
            print()
            print(f"Primer -- prvih 5 akcija u {demo_names[0]} (7 uglova zglobova + gripper):")
            print(np.round(data[demo_names[0]]["actions"][:5], 3))


def replay(path, demo_name, control_freq=20, camera_names=("agentview", "robot0_eye_in_hand", "sideview"), save_video=None):
    """
    Vizuelno pusta jednu demonstraciju preko RoboSuite viewer-a, I dodatno
    prikazuje snimljene kamere (agentview/eye_in_hand/sideview) u zasebnom
    cv2 prozoru -- env.render() (glavni RoboSuite prozor) je uvek SAMO
    JEDNA (interaktivna) kamera, ne prikazuje camera_names automatski.

    save_video (str ili None): ako je zadata putanja (npr. "replay.mp4"),
    ISTOVREMENO se snima video fajl -- ISTA slika (spojene kamere) koja se
    prikazuje uzivo preko cv2.imshow, preko cv2.VideoWriter. Ovo je
    pouzdanije od snimanja ekrana posebnim programom (nema rizika da neki
    drugi prozor/notifikacija upadne u kadar), i profesor ne mora nista
    da instalira da bi VIDEO pogledao (samo demo.hdf5/kod mu i dalje trebaju
    ako zeli da ga sam PUSTI, ne samo POGLEDA).

    Bazirano DIREKTNO na zvanicnom
    robosuite/scripts/playback_demonstrations_from_hdf5.py (proveren
    izvor), prosireno sa camera_names prikazom preko
    env._get_observations(force_update=True) -- isti poziv koji robosuite
    sam koristi interno da osvezi opservacije bez env.step() poziva.
    """
    import json

    import cv2
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"

    import robosuite as suite
    import hanoi_three_env  # noqa: F401 -- registruje "HanoiThree" u robosuite

    with h5py.File(path, "r") as f:
        env_info = json.loads(f["data"].attrs["env_info"])
        demo = f["data"][demo_name]
        model_xml = demo.attrs["model_file"]
        states = demo["states"][()]

    use_cameras = bool(camera_names)

    if save_video and not use_cameras:
        raise ValueError("save_video zahteva bar jednu kameru u camera_names -- nema sta da se snimi.")

    env = suite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=use_cameras,
        use_camera_obs=use_cameras,
        camera_names=list(camera_names) if use_cameras else None,
        camera_heights=256 if use_cameras else None,
        camera_widths=256 if use_cameras else None,
        ignore_done=True,
        control_freq=control_freq,
    )

    env.reset()
    xml = env.edit_model_xml(model_xml)  # NE postprocess_model_xml -- to je metoda na env-u, ne samostalna funkcija
    env.reset_from_xml_string(xml)
    env.sim.reset()
    if getattr(env, "viewer", None) is not None:
        env.viewer.set_camera(0)

    print(f"Puštam {demo_name} -- {len(states)} koraka...")
    if use_cameras:
        print(f"Dodatne kamere u zasebnom prozoru: {list(camera_names)}")
    if save_video:
        print(f"Snimam video u: {save_video}")

    video_writer = None  # inicijalizuje se na PRVI frejm, kad znamo tacnu sirinu/visinu spojene slike

    for state in states:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        if env.renderer == "mjviewer":
            env.viewer.update()
        env.render()

        if use_cameras:
            obs = env._get_observations(force_update=True)
            tiles = []
            for cam in camera_names:
                key = f"{cam}_image"
                if key in obs:
                    tiles.append(cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR))
            if tiles:
                combined = cv2.hconcat(tiles)
                cv2.imshow("Replay -- snimljene kamere", combined)
                cv2.waitKey(1)

                if save_video:
                    if video_writer is None:
                        h, w = combined.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(save_video, fourcc, control_freq, (w, h))
                    video_writer.write(combined)

        time.sleep(1.0 / control_freq)

    print("Gotovo.")
    if video_writer is not None:
        video_writer.release()
        print(f"Video sacuvan: {save_video}")
    if use_cameras:
        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Upotreba: python inspect_dataset.py putanja/do/demo.hdf5 [--replay demo_1] [--save_video video.mp4]")
        sys.exit(1)

    path = sys.argv[1]

    if "--replay" in sys.argv:
        idx = sys.argv.index("--replay")
        demo_name = sys.argv[idx + 1]

        save_video = None
        if "--save_video" in sys.argv:
            vidx = sys.argv.index("--save_video")
            save_video = sys.argv[vidx + 1]

        replay(path, demo_name, save_video=save_video)
    else:
        inspect(path)