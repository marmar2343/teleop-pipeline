"""
calibration.py

Kalibracija matrice rotacije izmedju MediaPipe (kamera) koordinatnog
sistema i robot base frame-a -- koristi POSTOJECI SPACE-clutch mehanizam
iz MediaPipeDevice, ne pravi novi UI.

v2 -- promenjena METODA HVATANJA na osnovu stvarnih kalibracionih podataka:
prva verzija je belezila NAJVECI pomeraj TOKOM pokreta ("max norm during
hold"), sto se pokazalo osetljivo na zakrivljenost putanje ruke (ljudska
ruka retko ide pravolinijski na kratkom pokretu) -- sva tri pravca su
zavrsila sa dominantnom Z-komponentom umesto ocekivanih razlicitih osa.

Sada se trazi da operater FIZICKI ZASTANE na krajnjoj tacki (drzi mirno
~0.5s) PRE otpustanja SPACE, i belezi se PROSEK tog mirnog perioda --
mnogo manje osetljivo na oblik putanje kojom je ruka stigla do te tacke.
Svaki pravac se dodatno hvata 2 puta (n_repeats) da odmah vidis koliko se
dva pokusaja slazu -- veliko razmimoilazenje je znak da ponovis taj pravac.

Postupak ostaje isti kao pre: za svaku od 3 robotske ose, drzi SPACE,
pomeri PRAVO do krajnje tacke, ZASTANI, pusti SPACE. Iz sva tri para
(kamera_smer <-> robot_smer) resava se Wahba/Kabsch problem (SVD) za
najbolju ORTONORMALNU rotacionu matricu R takvu da:

    robot_frame_delta = R @ kamera_frame_delta
"""

import time
from collections import deque

import numpy as np


def _wait_for_clutch_engage(device, poll_dt):
    """Blokira dok operater ne pritisne SPACE (aktivira clutch)."""
    while True:
        with device.lock:
            active = device._clutch_active
        if active:
            return
        time.sleep(poll_dt)


def _hold_and_average(device, hold_still_window, poll_dt):
    """
    Dok je SPACE drzan, sakuplja device.pos u kruzni bafer velicine
    hold_still_window sekundi. Vraca prosek TOG bafera u trenutku
    otpustanja SPACE-a -- pretpostavka je da operater zastane na krajnjoj
    tacki pre otpustanja, pa poslednji deo bafera odrazava mirno stanje
    na krajnjoj poziciji, ne celu (mozda zakrivljenu) putanju.

    Returns:
        (3,) prosek, ili None ako SPACE nije bio drzan dovoljno dugo da se
        sakupi bar jedan uzorak
    """
    buffer = deque(maxlen=max(1, int(hold_still_window / poll_dt)))
    while True:
        with device.lock:
            active = device._clutch_active
            current = device.pos.copy()
        if not active:
            break
        buffer.append(current)
        time.sleep(poll_dt)

    if len(buffer) == 0:
        return None
    return np.mean(buffer, axis=0)


def _capture_one_direction(device, label, min_displacement, hold_still_window, poll_dt):
    """
    Trazi od operatera JEDAN pokret za dati pravac. Umesto najveceg pomeraja
    TOKOM pokreta, belezi PROSEK poslednjih hold_still_window sekundi PRE
    otpustanja SPACE -- pretpostavka je da operater zastane na krajnjoj
    tacki pre nego sto pusti dugme.

    Returns:
        normalizovan (3,) vektor, ili None ako je pomeraj bio premali
        (operater treba da ponovi ovaj pokusaj)
    """
    print(f"\nPravac: {label}")
    print("Drzi SPACE, pomeri se PRAVO do krajnje tacke, ZASTANI tu bar pola sekunde, pa pusti SPACE.")
    print("Cekam SPACE...")
    _wait_for_clutch_engage(device, poll_dt)

    print("Pratim pokret...")
    captured = _hold_and_average(device, hold_still_window, poll_dt)

    if captured is None:
        print("  Nije zabelezen nijedan uzorak (prebrzo pusten SPACE) -- ponovi.")
        return None

    norm = np.linalg.norm(captured)
    if norm < min_displacement:
        print(
            f"  Pomeraj premali ({norm * 100:.1f}cm, treba bar {min_displacement * 100:.0f}cm) "
            f"-- ponovi, i probaj da zastanes JASNO na krajnjoj tacki."
        )
        return None

    print(f"  OK -- pomeraj {norm * 100:.1f}cm, smer {captured.round(4)}")
    return captured / norm


def calibrate_scale(device, true_distance_m, hold_still_window=0.5, n_repeats=3, poll_dt=0.01):
    """
    Meri sistematsku gresku MediaPipe hand_world_landmarks procene skale --
    koliko treba pomnoziti sirov MediaPipe pomeraj da bi odgovarao STVARNOJ
    fizickoj udaljenosti. MediaPipe world landmarks su naucena "metricka"
    procena bez pravog senzora dubine (monokularna kamera), pa sistematski
    gresi u apsolutnoj skali -- obicno potcenjuje, kao u tvom slucaju.

    VAZNO: postavi device.pos_sensitivity = 1.0 PRE ove kalibracije,
    inace mere kombinaciju stvarne MediaPipe greske i trenutnog
    pos_sensitivity faktora, ne cistu MediaPipe gresku. Rezultat ove
    funkcije treba da POSTANE nova pos_sensitivity vrednost (zameni staru,
    i ukloni odvojeni 'scale' iz test_teleoperation.py da ne mnozis dvaput
    na dva mesta).

    NAPOMENA o preciznosti: monokularna greska skale cesto zavisi i od
    stvarne udaljenosti ruke od kamere (nije nuzno identican faktor svuda
    u radnom prostoru) -- ovo daje JEDAN globalni faktor kao razumnu
    aproksimaciju, ne savrsenu korekciju za svaku pozu.

    Args:
        device: MediaPipeDevice instanca
        true_distance_m (float): TACNA fizicka udaljenost (metri) koju ces
            pomeriti ruku -- izmeri je unapred (lenjir/pantljika/markeri na
            stolu), ne nagadjaj
        hold_still_window, n_repeats, poll_dt: isto znacenje kao u
            calibrate_camera_to_robot_rotation

    Returns:
        scale (float): pomnozi sirov MediaPipe pomeraj sa ovim da dobijes
            priblizno tacnu fizicku distancu -- postavi kao device.pos_sensitivity
    """
    print("\n" + "=" * 60)
    print(f"KALIBRACIJA SKALE -- pomeri ruku TACNO {true_distance_m * 100:.0f}cm")
    print("Koristi lenjir/pantljiku/markere na stolu -- ne nagadjaj distancu.")
    print(f"Ponovices ovo {n_repeats}x.")
    print("=" * 60)

    measured_norms = []
    while len(measured_norms) < n_repeats:
        print(f"\n--- Ponavljanje {len(measured_norms) + 1}/{n_repeats} ---")
        print(f"Drzi SPACE, pomeri se TACNO {true_distance_m * 100:.0f}cm, ZASTANI, pusti SPACE.")
        print("Cekam SPACE...")
        _wait_for_clutch_engage(device, poll_dt)

        print("Pratim pokret...")
        captured = _hold_and_average(device, hold_still_window, poll_dt)

        if captured is None:
            print("  Nije zabelezen nijedan uzorak -- ponovi.")
            continue

        norm = np.linalg.norm(captured)
        print(f"  MediaPipe izmerio: {norm * 100:.1f}cm (ti si pomerila {true_distance_m * 100:.0f}cm)")
        measured_norms.append(norm)

    mean_measured = np.mean(measured_norms)
    std_measured = np.std(measured_norms)
    scale = true_distance_m / mean_measured

    print("\n" + "=" * 60)
    print(f"Prosecno izmereno: {mean_measured * 100:.1f}cm (std {std_measured * 100:.1f}cm) za stvarnih {true_distance_m * 100:.0f}cm")
    print(f"Faktor korekcije skale: {scale:.2f}x")
    if std_measured / mean_measured > 0.15:
        print("Upozorenje: velik razmak izmedju ponavljanja (>15%) -- probaj ponovo, sporije i pazljivije.")
    print("Postavi ovo kao device.pos_sensitivity, i ukloni odvojeni 'scale'")
    print("iz test_teleoperation.py da ne mnozis dvaput na dva mesta.")
    print("=" * 60 + "\n")

    return scale


def calibrate_camera_to_robot_rotation(
    device,
    axis_labels=None,
    min_displacement=0.04,
    hold_still_window=0.5,
    n_repeats=2,
    poll_dt=0.01,
):
    """
    Vodi operatera kroz kalibraciju preko postojeceg SPACE-clutch mehanizma.

    Args:
        device: MediaPipeDevice instanca (mora imati .lock, ._clutch_active, .pos)
        axis_labels (list od 3 stringa): opis svake robotske ose, redom
            [+X, +Y, +Z] u ROBOT frame-u.
        min_displacement (float): minimalan pomeraj (metri) da bi se
            pokusaj prihvatio. Podignuto na 4cm (sa prvobitnih 2cm) jer su
            se pokazali kraci pokreti kao nepouzdani -- forsira jasniji,
            namerniji pokret.
        hold_still_window (float): koliko sekundi pred kraj drzanja SPACE
            se prosecira za konacnu vrednost (pretpostavka: operater
            zastane na krajnjoj tacki pre otpustanja)
        n_repeats (int): koliko puta se svaki pravac hvata nezavisno --
            prosek ponavljanja se koristi kao konacan smer, i ispisuje se
            razmimoilazenje izmedju ponavljanja radi kontrole kvaliteta
        poll_dt (float): koliko cesto (sekunde) se proverava device stanje

    Returns:
        R (3,3): rotaciona matrica takva da je
            robot_frame_delta = R @ kamera_frame_delta
    """
    if axis_labels is None:
        axis_labels = ["+X (napred, dalje od tela)", "+Y (ulevo)", "+Z (nagore)"]

    assert len(axis_labels) == 3, "Treba tacno 3 ose za kalibraciju rotacije u 3D"

    camera_dirs = []

    print("\n" + "=" * 60)
    print("KALIBRACIJA KAMERA -> ROBOT ROTACIJE")
    print("Za svaki korak: drzi SPACE, pomeri se PRAVO do krajnje tacke,")
    print("ZASTANI tu bar pola sekunde, pa pusti SPACE. Svaki pravac")
    print(f"hvatas {n_repeats}x nezavisno.")
    print("=" * 60)

    for i, label in enumerate(axis_labels):
        reps = []
        while len(reps) < n_repeats:
            print(f"\n--- Korak {i + 1}/3: {label} (ponavljanje {len(reps) + 1}/{n_repeats}) ---")
            vec = _capture_one_direction(device, label, min_displacement, hold_still_window, poll_dt)
            if vec is not None:
                reps.append(vec)

        if n_repeats > 1:
            spread = [
                np.degrees(np.arccos(np.clip(np.dot(reps[0], r), -1.0, 1.0)))
                for r in reps[1:]
            ]
            spread_str = ", ".join(f"{s:.1f}" for s in spread)
            flag = "  <- prilicno razlicito, razmisli da ponovis ceo ovaj pravac" if any(s > 15 for s in spread) else ""
            print(f"  Razmimoilazenje izmedju ponavljanja: {spread_str} stepeni{flag}")

        avg_vec = np.mean(reps, axis=0)
        avg_vec = avg_vec / np.linalg.norm(avg_vec)
        camera_dirs.append(avg_vec)

    # -- provera KVALITETA kalibracije, ne samo da li je R validna rotacija --
    # (SVD ce UVEK vratiti validnu rotaciju, cak i od losih/mesanih pokreta)
    P = np.array(camera_dirs)  # (3,3), red i = kamera-smer za robot-osu i
    Q = np.eye(3)              # red i = e_i

    print("\nUglovi izmedju tvoja tri (usrednjena) pravca (idealno ~90 stepeni):")
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        cos_angle = np.clip(np.dot(P[a], P[b]), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        flag = "  <- daleko od 90, razmisli da ponovis ovaj par pravaca" if abs(angle - 90) > 20 else ""
        print(f"  pravac {a + 1} <-> pravac {b + 1}: {angle:.1f} stepeni{flag}")

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    print("\n" + "=" * 60)
    print("Kalibracija zavrsena. Matrica R:")
    print(R.round(4))
    print("=" * 60 + "\n")

    return R