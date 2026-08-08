"""
ik_solver.py

DLS_IK_Solver -- diferencijalna IK (damped least squares), JEDAN korak po
pozivu (resolved-rate pristup, videti objasnjenje u razgovoru).

Ovo je spoj dva pristupa:
  - struktura/keširanje (klasa, eef_id/joint indeksi se racunaju JEDNOM u
    __init__, ne na svaki poziv step()) -- prema predlogu iz razgovora
  - sigurnosne granice (max_joint_step, ogranicenje na jnt_range) i
    koriscenje robosuite-ovog sim.data.get_site_jacp()/get_site_jacr()
    umesto sirovog mujoco.mj_jacSite(sim.model, sim.data, ...)

VAZNA NAPOMENA o poslednjoj tacki: sim.model / sim.data u robosuite-u NISU
sirovi mujoco.MjModel/mujoco.MjData objekti -- to su robosuite wrapper
klase (robosuite/utils/binding_utils.py: MjModel/MjData), koje pravi
mujoco.MjModel/MjData drze unutra kao self._model / self._data. Sirova
mujoco.mj_jacSite(model, data, ...) funkcija ocekuje TACNO te sirove
objekte, pa poziv mujoco.mj_jacSite(sim.model, sim.data, ...) (bez
._model/._data) skoro sigurno baca TypeError. Zato ovde koristimo
sim.data.get_site_jacp(name)/get_site_jacr(name) -- robosuite-ove vec
gotove wrapper metode koje iznutra ispravno pozivaju mujoco.mj_jacSite
sa ._model/._data.

POZICIJA + OPCIONA FIKSNA ORIJENTACIJA. Ako je orijentacija "zakljucana"
(lock_current_orientation()), IK koristi 6-DOF Jakobijan (pozicija + rotacija)
i aktivno vraca hvataljku na zapamcenu orijentaciju svaki put kad null-space
kretanje pokusa da je promeni -- videti DLS_IK_Solver.step().
"""

import numpy as np
import robosuite.utils.transform_utils as T


class DLS_IK_Solver:
    """
    Damped least squares diferencijalna IK za JOINT_POSITION kontroler.

    Args:
        env: robosuite environment (radi i kroz VisualizationWrapper, jer
            on prosledjuje .sim/.robots atribute ka pravom environment-u)
        arm (str): "right" za jednorucne robote (npr. Panda)
        damping (float): DLS prigusenje -- vece = stabilnije blizu
            singulariteta (npr. ispruzena ruka), ali sporija konvergencija.
            Eksperimentalni parametar, 0.05 je samo razuman pocetak.
        step_size (float): faktor skaliranja jednog DLS koraka (0-1) --
            manje od 1.0 usporava/stabilizuje konvergenciju
        max_joint_step (float): najveci dozvoljen pomeraj JEDNOG zgloba
            (rad) po pozivu -- sigurnosna kocnica protiv naglih skokova
            (npr. ako MediaPipe na trenutak izgubi pa naglo ponovo nadje ruku)
        clip_to_limits (bool): ako True, rezultat se dodatno ogranicava na
            fizicke granice zglobova robota (sim.model.jnt_range)
        lock_orientation (bool): ako True (podrazumevano), ORIJENTACIJA
            hvataljke u trenutku konstrukcije se odmah "zakljucava" kao
            fiksni cilj -- IK ce od tog trenutka aktivno sprecavati rotaciju
            hvataljke, ne samo pratiti poziciju. Pozovi lock_current_orientation()
            rucno kasnije ako zelis da "prezakljucas" na neku NOVU orijentaciju
            (npr. posle rucne rotacije nekim drugim mehanizmom).
    """

    def __init__(
        self,
        env,
        arm="right",
        damping=0.05,
        step_size=0.5,
        max_joint_step=0.2,
        clip_to_limits=True,
        lock_orientation=True,
    ):
        self.env = env
        self.sim = env.sim
        self.arm = arm
        self.damping = damping
        self.step_size = step_size
        self.max_joint_step = max_joint_step
        self.clip_to_limits = clip_to_limits

        robot = env.robots[0]

        # -- sve sto se ne menja iz koraka u korak, izracunato JEDNOM ovde --
        self.eef_site_id = robot.eef_site_id[arm]
        self.eef_site_name = self.sim.model.site_id2name(self.eef_site_id)

        joint_names = robot.robot_model.joints
        self.qpos_idx = np.array([self.sim.model.get_joint_qpos_addr(n) for n in joint_names])
        self.qvel_idx = np.array([self.sim.model.get_joint_qvel_addr(n) for n in joint_names])

        if clip_to_limits:
            joint_ids = np.array([self.sim.model.joint_name2id(n) for n in joint_names])
            self.joint_limits = self.sim.model.jnt_range[joint_ids].copy()
        else:
            self.joint_limits = None

        self.target_orientation = None  # (x,y,z,w) kvaternion, ili None = ne kontrolisi orijentaciju
        if lock_orientation:
            self.lock_current_orientation()

    def get_eef_position(self):
        """Trenutna (stvarna, iz simulacije) pozicija hvataljke."""
        return np.copy(self.sim.data.site_xpos[self.eef_site_id])

    def get_eef_orientation(self):
        """Trenutna orijentacija hvataljke kao kvaternion (x,y,z,w)."""
        mat = self.sim.data.site_xmat[self.eef_site_id].reshape(3, 3)
        return T.mat2quat(mat)

    def lock_current_orientation(self):
        """
        Zapamti TRENUTNU orijentaciju hvataljke kao fiksni cilj -- od ovog
        poziva nadalje, step() ce aktivno vracati orijentaciju na ovu
        vrednost (preko 6-DOF Jakobijana), ne samo pratiti poziciju.

        Pozovi ovo kad god zelis da "prezakljucas" na novu orijentaciju
        (npr. na pocetku svake nove clutch sesije, ako zelis da svaka
        sesija pocinje od ONOGA STO ROBOT TRENUTNO IMA, a ne uvek od iste
        pocetne orijentacije).
        """
        self.target_orientation = self.get_eef_orientation()

    def get_current_qpos(self):
        """Trenutni STVARNI joint uglovi ruke iz simulacije."""
        return np.copy(self.sim.data.qpos[self.qpos_idx])

    def get_jacobian(self):
        """
        Translacioni Jakobijan (3 x n_zglobova), samo kolone za zglobove
        NASE ruke -- sim.data.get_site_jacp() vraca pun (3 x nv) Jakobijan
        za CEO model (robot + kocke + gripper), pa se ovde iseca samo ono
        sto nam treba preko self.qvel_idx.
        """
        jacp_full = self.sim.data.get_site_jacp(self.eef_site_name).reshape(3, -1)
        return jacp_full[:, self.qvel_idx]

    def get_jacobian_rot(self):
        """Rotacioni Jakobijan (3 x n_zglobova), analogno get_jacobian()."""
        jacr_full = self.sim.data.get_site_jacr(self.eef_site_name).reshape(3, -1)
        return jacr_full[:, self.qvel_idx]

    def step(self, q_current, target_position):
        """
        Jedan DLS IK korak, sa PRIORITIZOVANOM (null-space) kombinacijom
        pozicije i orijentacije.

        Args:
            q_current (n,): trenutni komandovani uglovi -- najbezbednije je
                proslediti self.get_current_qpos() (stvarno stanje simulacije).
                Ako umesto toga pratis sopstveni "komandovani" niz uglova
                nezavisno od simulacije (da izbegnes lag interpolatora),
                povremeno ga resetuj na get_current_qpos() da se razmak
                izmedju njih ne akumulira -- videti napomenu iz razgovora.
            target_position (3,): zeljena APSOLUTNA pozicija hvataljke
                (world frame, isti sistem kao sim.data.site_xpos)

        Returns:
            q_new (n,): predlozeni sledeci joint uglovi -- saljes ih
                DIREKTNO kao akciju JOINT_POSITION kontroleru
                (input_type="absolute")

        NAPOMENA o orijentaciji: ako je self.target_orientation postavljen
        (podrazumevano, preko lock_orientation=True u konstruktoru), IK
        AUTOMATSKI dodaje orijentacionu gresku pored pozicione, kao JEDAN
        spojen 6D sistem (pozicija + rotacija zajedno u istom linearnom
        resenju). Probala sam i null-space prioritizovan pristup (pozicija
        kao "glavni" zadatak, orijentacija ogranicena da ne remeti poziciju)
        -- IZMERENO LOSIJE za ovaj slucaj (7 zglobova, 6 zadataka, samo 1
        redundantan DOF): spojeni sistem prirodno nalazi resenje koje dobro
        zadovoljava OBA zadatka, dok null-space pristup previse ogranicava
        orijentacionu korekciju na ono malo slobodnog prostora sto ostaje.
        Zadrzano na spojenom pristupu na osnovu stvarnog testa, ne teorije.
        """
        current_position = self.get_eef_position()
        err_pos = np.array(target_position, dtype=float) - current_position
        Jp = self.get_jacobian()

        lam_sq = self.damping ** 2

        if self.target_orientation is not None:
            current_orn = self.get_eef_orientation()
            err_rot = T.get_orientation_error(self.target_orientation, current_orn)
            Jr = self.get_jacobian_rot()

            J = np.vstack([Jp, Jr])                     # (6, n_zglobova)
            error = np.concatenate([err_pos, err_rot])   # (6,)
            dq = J.T @ np.linalg.solve(J @ J.T + lam_sq * np.eye(6), error)
        else:
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lam_sq * np.eye(3), err_pos)

        dq = np.clip(self.step_size * dq, -self.max_joint_step, self.max_joint_step)

        q_new = q_current + dq

        if self.clip_to_limits:
            q_new = np.clip(q_new, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return q_new