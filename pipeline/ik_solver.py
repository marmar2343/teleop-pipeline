"""
ik_solver.py

DLS_IK_Solver - differential inverse kinematics using damped least squares.
Performs one step per call (resolved-rate approach).

The solver uses a resolved-rate approach and computes one joint-space update
per call to step(). It supports both position-only and combined
position-orientation control, together with joint-step and joint-limit
constraints.
"""

import numpy as np
import robosuite.utils.transform_utils as T


class DLS_IK_Solver:
    """
    Damped Least Squares differential IK for a JOINT_POSITION controller.

    Args:
        env: Robosuite environment.
        arm (str): Arm identifier, e.g. "right".
        damping (float): DLS damping coefficient. Larger values improve numerical 
        stability near singular configurations, at the cost of slower convergence.
        step_size (float): Scaling factor applied to each IK update.
            Values below 1.0 produce smaller and smoother updates.
        max_joint_step (float): Maximum allowed change of an individual
            joint angle per IK step, in radians.
        clip_to_limits (bool): If True, the resulting joint positions are
            constrained to the robot's physical joint limits.
        lock_orientation (bool): If True, the current end-effector
            orientation is stored as a fixed target during initialization.
    """

    def __init__(self, env, arm="right", damping=0.05, step_size=0.5, max_joint_step=0.2,
        clip_to_limits=True, lock_orientation=True,):
        self.env = env
        self.sim = env.sim
        self.arm = arm
        self.damping = damping
        self.step_size = step_size
        self.max_joint_step = max_joint_step
        self.clip_to_limits = clip_to_limits

        robot = env.robots[0]

        # cache the end-effector site information used by the solver
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

        # Fixed orientation target; None means orientation is not controlled.
        self.target_orientation = None  

        if lock_orientation:
            self.lock_current_orientation()

    def get_eef_position(self):
        # return the current end-effector position in the world frame
        return np.copy(self.sim.data.site_xpos[self.eef_site_id])

    def get_eef_orientation(self):
        # return the current end-effector orientation as an (x, y, z, w) quaternion
        mat = self.sim.data.site_xmat[self.eef_site_id].reshape(3, 3)
        return T.mat2quat(mat)

    def lock_current_orientation(self):
        # store the current end-effector orientation as a fixed target
        #
        # the target is subsequently used to compute the orientation error in the 6-DOF IK task
        self.target_orientation = self.get_eef_orientation()

    def get_current_qpos(self):
        # return the current joint positions of the robot arm
        return np.copy(self.sim.data.qpos[self.qpos_idx])

    def get_jacobian(self):
        # return the translational end-effector Jacobian

        jacp_full = self.sim.data.get_site_jacp(self.eef_site_name).reshape(3, -1)
        return jacp_full[:, self.qvel_idx]

    def get_jacobian_rot(self):
        # return the rotational end-effector Jacobian
        jacr_full = self.sim.data.get_site_jacr(self.eef_site_name).reshape(3, -1)
        return jacr_full[:, self.qvel_idx]

    def step(self, q_current, target_position):
        """
        Perform one DLS IK update.

        Args:
            q_current (n,): Current joint positions of the robot arm.
            target_position (3,): Desired absolute end-effector position
                expressed in the world frame.

        Returns:
            q_new (n,): Updated joint positions to be sent to the
                JOINT_POSITION controller.

        If a target orientation is defined, position and orientation are
        solved simultaneously using a combined 6-DOF Jacobian.
        """
        # compute the Cartesian position error
        current_position = self.get_eef_position()
        err_pos = np.array(target_position, dtype=float) - current_position

        # obtain the translational Jacobian for the selected arm
        Jp = self.get_jacobian()

        lam_sq = self.damping ** 2

        if self.target_orientation is not None:
            # compute the orientation error and rotational Jacobian
            current_orn = self.get_eef_orientation()
            err_rot = T.get_orientation_error(self.target_orientation, current_orn)
            Jr = self.get_jacobian_rot()

            J = np.vstack([Jp, Jr])
            error = np.concatenate([err_pos, err_rot])  

            # DLS
            # dq = J^T (J J^T + lambda^2 I)^(-1) error
            dq = J.T @ np.linalg.solve(J @ J.T + lam_sq * np.eye(6), error)
        else:
            # position-only DLS IK
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lam_sq * np.eye(3), err_pos)

        # scale the update and limit the max joint displacement
        dq = np.clip(self.step_size * dq, -self.max_joint_step, self.max_joint_step)

        # apply the joint space update
        q_new = q_current + dq

        if self.clip_to_limits:
            # make sure that all joints remain within their physical limits
            q_new = np.clip(q_new, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return q_new