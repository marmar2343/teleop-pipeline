"""
hanoi_three_env.py

Custom RoboSuite environment implementing a three-cube Tower of Hanoi task.

The task uses three cubes instead of conventional disks with holes. The
cubes are stacked on one of three logical peg regions on the table rather
than physically mounted onto pegs.

The initial state contains the complete three-cube tower on the source peg.
The objective is to transfer the entire tower to the target peg while
preserving the required size ordering.

The third peg remains available as an auxiliary location that the operator
can use when solving the task.

The environment uses sparse rewards by default, with optional partial-credit
reward shaping. This is suitable for recording human teleoperation
demonstrations, where the operator solves the task directly rather than
following a predefined optimal sequence.

Peg locations are represented by thin visual fiducial markers placed on the
table. These markers are visual-only objects and do not participate in the
physics simulation. Source and target identities can optionally be
communicated through marker colors.
"""

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


class HanoiThree(ManipulationEnv):
    """
    Three-cube Tower of Hanoi environment with simplified cube geometry.

    The cubes are ordered by size:
        cubeA: largest and bottom cube
        cubeB: medium cube
        cubeC: smallest and top cube

    Args:
        robots (str or list of str): Robot model used by the environment.
            The task expects a single-arm robot, such as "Panda".
        controller_configs (dict): Robot controller configuration.
        source_peg_idx (int): Initial peg containing the complete tower.
            Must be 0, 1, or 2.
        target_peg_idx (int): Destination peg for the complete tower.
            Must differ from source_peg_idx.
        randomize_pegs (bool): If True, source and target pegs are randomly
            selected on every reset. If False, the constructor values are
            preserved.
        peg_spacing (float): Distance between neighboring peg centers in meters.
        reward_shaping (bool): If True, partial progress receives reward.
            If False, only a completed task receives a non-zero reward.
        color_code_pegs (bool): If True, source and target pegs are visually
            distinguished by color.
        Other parameters follow the standard RoboSuite environment interface.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        source_peg_idx=0,
        target_peg_idx=2,
        randomize_pegs=False,
        peg_spacing=0.18,
        color_code_pegs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        assert source_peg_idx != target_peg_idx,  "source_peg_idx and target_peg_idx must be different"
        assert source_peg_idx in (0, 1, 2) and target_peg_idx in (0, 1, 2), "Peg indices must be 0, 1, or 2"


        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        # define the three peg centers along the table's y-axis
        # the offsets are expressed relative to the table center
        self.peg_spacing = peg_spacing
        self.peg_offsets = {
            0: np.array([0.0, -peg_spacing]),
            1: np.array([0.0, 0.0]),
            2: np.array([0.0, peg_spacing]),
        }
        self.source_peg_idx = source_peg_idx
        self.target_peg_idx = target_peg_idx
        self.randomize_pegs = randomize_pegs
        self.color_code_pegs = color_code_pegs

        # half-sizes of the cubic objects in meters
        # the decreasing dimensions enforce the intended size hierarchy        
        self.half_heights = {"cubeA": 0.025, "cubeB": 0.020, "cubeC": 0.015}

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # Geometric and state checks

    def _peg_world_xy(self, peg_idx):
        """Return the absolute world-frame (x, y) position of a peg."""
        return self.table_offset[:2] + self.peg_offsets[peg_idx]

    def _cube_at_peg(self, cube, peg_idx, xy_tol=0.03):
        """Return True if a cube is within xy_tol of the specified peg."""
        pos_xy = np.array(self.sim.data.body_xpos[self.obj_body_id[cube.name]])[:2]
        return np.linalg.norm(pos_xy - self._peg_world_xy(peg_idx)) < xy_tol

    def _b_on_a(self):
        """Return True if cubeB is resting on cubeA and is not grasped."""
        grasping_b = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeB)
        return (not grasping_b) and self.check_contact(self.cubeB, self.cubeA)

    def _c_on_b(self):
        """Return True if cubeC is resting on cubeB and is not grasped."""
        grasping_c = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)
        return (not grasping_c) and self.check_contact(self.cubeC, self.cubeB)


    # Reward
    def reward(self, action):
        """
        Compute the task reward.

        Partial progress is evaluated using the following structure:

            +1.0  cubeA reaches the target peg
            +1.0  cubeB is correctly placed on cubeA
            +2.0  cubeC is correctly placed on cubeB

        Therefore, a fully reconstructed tower receives a total reward of
        4.0 before reward scaling.

        When reward_shaping is disabled, the reward is sparse: zero is
        returned until the complete tower is correctly placed at the target.
        """
        reward = 0.0
        if self._cube_at_peg(self.cubeA, self.target_peg_idx):
            reward += 1.0
            if self._b_on_a():
                reward += 1.0
                if self._c_on_b():
                    reward += 2.0

        if not self.reward_shaping:
            reward = 4.0 if reward >= 4.0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 4.0

        return reward

    # Scene construction
    def _load_model(self):
        """
        Construct the table, robot, cubes, and visual peg markers.

        Cube positions are initialized manually during reset so that the
        three cubes always form a complete tower at the beginning of an
        episode.
        """
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
            tex_attrib=tex_attrib, mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen", tex_name="greenwood", mat_name="greenwood_mat",
            tex_attrib=tex_attrib, mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue", tex_name="bluewood", mat_name="bluewood_mat",
            tex_attrib=tex_attrib, mat_attrib=mat_attrib,
        )

        h = self.half_heights
        self.cubeA = BoxObject(
            name="cubeA", size_min=[h["cubeA"]] * 3, size_max=[h["cubeA"]] * 3,
            rgba=[1, 0, 0, 1], material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB", size_min=[h["cubeB"]] * 3, size_max=[h["cubeB"]] * 3,
            rgba=[0, 1, 0, 1], material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC", size_min=[h["cubeC"]] * 3, size_max=[h["cubeC"]] * 3,
            rgba=[0, 0, 1, 1], material=bluewood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC]

        # Visual-only markers represent the three logical peg regions.
        # They are static and non-colliding, so they provide spatial
        # references without affecting the simulated dynamics.
        self.peg_markers = []
        marker_z = self.table_offset[2] + 0.001  # tik iznad povrsine stola
        for idx, offset in self.peg_offsets.items():
            marker = CylinderObject(
                name=f"peg{idx}_marker",
                size=[0.045, 0.001],  
                rgba=[0.25, 0.25, 0.25, 0.6],  
                joints=None,
                obj_type="visual",)
            marker_pos = np.array(
                [self.table_offset[0] + offset[0], self.table_offset[1] + offset[1], marker_z])
            marker.get_obj().set("pos", array_to_string(marker_pos))
            self.peg_markers.append(marker)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes + self.peg_markers,)

    def _setup_references(self):
        """
        Cache MuJoCo body and geometry IDs used during simulation.

        Caching these identifiers avoids repeatedly resolving object and
        marker names during every simulation step.
        """
        super()._setup_references()
        self.obj_body_id = {
            "cubeA": self.sim.model.body_name2id(self.cubeA.root_body),
            "cubeB": self.sim.model.body_name2id(self.cubeB.root_body),
            "cubeC": self.sim.model.body_name2id(self.cubeC.root_body),
        }

        # Store the MuJoCo geometry ID of each visual peg marker so that
        # marker colors can be updated efficiently during reset.
        self.peg_marker_geom_ids = {}
        for idx, marker in enumerate(self.peg_markers):
            geom_name = marker.visual_geoms[0] if marker.visual_geoms else marker.contact_geoms[0]
            self.peg_marker_geom_ids[idx] = self.sim.model.geom_name2id(geom_name)

    def _reset_internal(self):
        """
        Initialize the complete tower on the selected source peg.

        A small common xy perturbation is applied to the entire tower so
        that the initial configuration is not perfectly identical across
        randomized resets while keeping all three cubes aligned.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            if self.randomize_pegs:
                src, tgt = self.rng.choice(3, size=2, replace=False)
                self.source_peg_idx, self.target_peg_idx = int(src), int(tgt)

            print(f"[HanoiThree] source peg: {self.source_peg_idx}, "f"target peg: {self.target_peg_idx}")

            if self.color_code_pegs:
                for idx in range(3):
                    if idx == self.source_peg_idx:
                        rgba = [0.9, 0.55, 0.1, 0.8]    # orange identifies the current source peg
                    elif idx == self.target_peg_idx:
                        rgba = [0.15, 0.75, 0.15, 0.8]   # orange identifies the current source peg
                    else:
                        rgba = [0.25, 0.25, 0.25, 0.6]  # gray identifies the auxiliary peg
                    self.sim.model.geom_rgba[self.peg_marker_geom_ids[idx]] = rgba

            # Apply the same xy perturbation to all cubes so that the tower remains vertically aligned after randomization.
            xy = self._peg_world_xy(self.source_peg_idx) + self.rng.uniform(-0.005, 0.005, size=2)

            identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z), without rotation
            z_cursor = self.table_offset[2] 

            for name in ["cubeA", "cubeB", "cubeC"]:
                half_h = self.half_heights[name]
                z_center = z_cursor + half_h
                obj = getattr(self, name)
                pos = np.array([xy[0], xy[1], z_center])
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, identity_quat]))
                z_cursor = z_center + half_h  # The top of the current cube becomes the base height for the next cube in the tower

    def _setup_observables(self):
        """
        Configure object-related observations.

        For each cube, the observation space includes its world-frame
        position and orientation. The target peg position and relative
        end-effector-to-object positions are also exposed for downstream
        control and learning applications.
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"
            cube_names = ["cubeA", "cubeB", "cubeC"]

            sensors = []
            for name in cube_names:
                @sensor(modality=modality)
                def cube_pos(obs_cache, obj_name=name):
                    return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

                cube_pos.__name__ = f"{name}_pos"

                @sensor(modality=modality)
                def cube_quat(obs_cache, obj_name=name):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.obj_body_id[obj_name]]), to="xyzw")

                cube_quat.__name__ = f"{name}_quat"

                sensors += [cube_pos, cube_quat]

            # Expose the current target peg position so that the target
            # location remains observable when peg identities are randomized.
            @sensor(modality=modality)
            def target_peg_pos(obs_cache):
                xy = self._peg_world_xy(self.target_peg_idx)
                return np.array([xy[0], xy[1], self.table_offset[2]])

            sensors.append(target_peg_pos)

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors += [
                self._get_obj_eef_sensor(full_pf, f"{cube}_pos", f"{arm_pf}gripper_to_{cube}", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                for cube in cube_names
            ]

            names = [s.__name__ for s in sensors]
            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables


    # Success and visualization
    def _check_success(self):
        """
        Return True when the complete tower is correctly placed on the target.

        Success requires:
            1. cubeA to be located on the target peg,
            2. cubeB to be resting on cubeA,
            3. cubeC to be resting on cubeB.
        """
        return (
            self._cube_at_peg(self.cubeA, self.target_peg_idx)
            and self._b_on_a()
            and self._c_on_b())

    def check_hanoi_legality(self, xy_tol=0.03):
        """
        Validate the current scene against the Tower of Hanoi constraints.

        Two conditions are checked:

            1. Every cube that is not currently grasped must be located
               within the valid region of one of the three pegs.

            2. Whenever multiple cubes occupy the same peg, they must be
               stacked in decreasing size order from bottom to top.

        A cube currently held by the gripper is treated as an intermediate
        state and is therefore excluded from the placement checks.

        Returns:
            tuple:
                is_legal (bool): Whether the current scene satisfies the
                    defined task constraints.
                reason (str or None): Description of the first detected
                    violation, or None when the state is legal.
        """
        cubes = {"cubeA": self.cubeA, "cubeB": self.cubeB, "cubeC": self.cubeC}
        size_rank = {"cubeA": 3, "cubeB": 2, "cubeC": 1}  

        grasped_name = None

        for name, cube in cubes.items():
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=cube):
                grasped_name = name
                break

        peg_contents = {0: [], 1: [], 2: []}

        for name, cube in cubes.items():
            if name == grasped_name:
                # a grasped cube is temporarily in transit and is not subject to peg-placement constraints
                continue 

            pos = self.sim.data.body_xpos[self.obj_body_id[name]]
            found_peg = None
            for idx in range(3):
                if np.linalg.norm(pos[:2] - self._peg_world_xy(idx)) < xy_tol:
                    found_peg = idx
                    break

            if found_peg is None:
                return (False, f"{name} is not located on any of the three pegs",)

            peg_contents[found_peg].append((name, pos[2]))

        # check the vertical ordering of cubes on every occupied peg
        for idx, items in peg_contents.items():
            if len(items) < 2:
                continue
            items_by_height = sorted(items, key=lambda t: t[1]) 
            for i in range(len(items_by_height) - 1):
                lower_name = items_by_height[i][0]
                upper_name = items_by_height[i + 1][0]
                if size_rank[lower_name] < size_rank[upper_name]:
                    return False, (
                        f"On peg {idx}, {upper_name} (larger) is "
                        f"above {lower_name} (smaller)"
                    )

        return True, None

    def visualize(self, vis_settings):
        """
        Add the currently relevant cube to RoboSuite's gripper visualization.
        """
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:

            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeC)


# example execution 
if __name__ == "__main__":
    import time

    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(controller="BASIC")

    env = robosuite.make(
        "HanoiThree",
        robots="Panda",
        controller_configs=controller_config,
        source_peg_idx=0,
        target_peg_idx=2,
        randomize_pegs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
        seed=42,
    )

    obs = env.reset()
    print("Observation keys:", list(obs.keys()))
    print(
        f"Source peg: {env.source_peg_idx}, "
        f"target peg: {env.target_peg_idx}"
    )


    for step in range(100):
        # Placeholder action for basic environment validation
        action = np.zeros(env.action_dim)  
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(1.0 / env.control_freq)
        if step % 20 == 0:
            print(f"step {step}: reward={reward:.3f}, success={env._check_success()}")
        if done:
            obs = env.reset()

    input("Press Enter to close...")
    env.close()