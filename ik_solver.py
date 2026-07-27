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

        # sve sto se ne menja iz koraka u korak, izracunato jednom ovde
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

        self.target_orientation = self.get_eef_orientation()

    def get_current_qpos(self):
        """Trenutni STVARNI joint uglovi ruke iz simulacije."""
        return np.copy(self.sim.data.qpos[self.qpos_idx])

    def get_jacobian(self):
    
        jacp_full = self.sim.data.get_site_jacp(self.eef_site_name).reshape(3, -1)
        return jacp_full[:, self.qvel_idx]

    def get_jacobian_rot(self):

        jacr_full = self.sim.data.get_site_jacr(self.eef_site_name).reshape(3, -1)
        return jacr_full[:, self.qvel_idx]

    def step(self, q_current, target_position):
        """
        Jedan DLS IK korak.

        NAPOMENA o orijentaciji: ako je self.target_orientation postavljen
        (podrazumevano, preko lock_orientation=True u konstruktoru), IK
        AUTOMATSKI dodaje orijentacionu gresku pored pozicione -- ne treba
        nista posebno prosledjivati, samo pozovi step() kao i pre. Ako
        NIKAD nisi pozvala lock_current_orientation() i konstruisala si sa
        lock_orientation=False, ponasanje je identicno staroj (samo-pozicija)
        verziji.
        """
        current_position = self.get_eef_position()
        err_pos = np.array(target_position, dtype=float) - current_position
        Jp = self.get_jacobian()

        if self.target_orientation is not None:
            current_orn = self.get_eef_orientation()
            err_rot = T.get_orientation_error(self.target_orientation, current_orn)
            Jr = self.get_jacobian_rot()

            J = np.vstack([Jp, Jr])          # (6, n_zglobova)
            error = np.concatenate([err_pos, err_rot])  # (6,)
        else:
            J = Jp
            error = err_pos

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * error
        # np.linalg.solve umesto eksplicitnog np.linalg.inv() -- numericki
        # stabilnije i brze, ne racuna se puna inverzija koja nam ionako
        # ne treba (samo nam treba resenje jednog linearnog sistema)
        lam_sq = self.damping ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam_sq * np.eye(J.shape[0]), error)
        dq = np.clip(self.step_size * dq, -self.max_joint_step, self.max_joint_step)

        q_new = q_current + dq

        if self.clip_to_limits:
            q_new = np.clip(q_new, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return q_new