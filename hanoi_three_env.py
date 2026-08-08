"""
hanoi_three_env.py

Custom RoboSuite okruzenje: PRAVI Hanoj zadatak (premestanje) sa 3 kocke
umesto diskova sa rupom -- kocke se ne "navlace" na klin, nego se slazu
na jednu od tri logicke zone (peg) na stolu.

Kljucna razlika u odnosu na prvu (pojednostavljenu) verziju stack_three_env.py:
    - Pocetno stanje NIJE nasumicno razbacano -- kula od sve tri kocke je
      VEC slozena na izvornom klinu (source_peg_idx), tacno kao u pravom
      Hanoju.
    - Cilj je preneti CELU kulu (ukljucujuci najvecu/donju kocku) na ciljni
      klin (target_peg_idx), ispravnim redosledom (najveca na dnu).
    - Treci klin ostaje slobodan kao prirodno pomocno mesto -- operater ga
      koristi po potrebi, ne mora se posebno programirati u environment-u.

Namerno NEMA guste (dense) nagrade koja vodi kroz optimalno resenje od
7 poteza -- za snimanje ljudskih demonstracija (teleoperacija) to nije
potrebno, operater sam resava fizicki zadatak gledajuci scenu. Nagrada je
uglavnom retka (sparse), sa blagim delimicnim bonusom (partial credit) koji
moze da posluzi i za eventualno RL poredjenje kasnije.

Klinovi su vizuelno obelezeni na stolu -- tanki, ravni "fiducial" markeri
(CylinderObject sa joints=None, obj_type="visual") koji ne ucestvuju u
fizici (nema kolizije), isti obrazac koji sam RoboSuite koristi za svoje
vizuelne fiducijale (npr. MilkVisualObject/BreadVisualObject u PickPlace
zadatku). Namerno su SVI markeri iste (neutralne) boje -- koji je klin
"cilj" treba da nosi jezicka instrukcija (za VLA fine-tuning), ne boja,
jer boja kao signal ne postoji na pravom stolu van simulacije.
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
    Premestanje kule od tri kocke (opadajuce velicine) sa izvornog na
    ciljni klin -- Hanojske kule sa pojednostavljenom (kocka umesto diska
    sa rupom) geometrijom.

    Args:
        robots (str or list of str): npr. "Panda". Mora biti jedan
            single-arm robot.
        controller_configs (dict): konfiguracija kontrolera.
        source_peg_idx (int): indeks klina (0, 1 ili 2) na kome je kula
            slozena na pocetku.
        target_peg_idx (int): indeks klina na koji kula treba da se prenese.
            Mora biti razlicit od source_peg_idx. Treci (neiskoriscen) klin
            je automatski dostupan operateru kao pomocno mesto.
        randomize_pegs (bool): ako je True, source/target se biraju
            nasumicno (razliciti) pri svakom reset()-u -- korisno za
            raznovrsnost demonstracija posto je pipeline vec proveren.
            Ako je False, koriste se fiksne vrednosti iz konstruktora
            (lakse za debug).
        peg_spacing (float): rastojanje izmedju susednih klinova (m).
        reward_shaping (bool): True = blagi "partial credit" tokom zadatka,
            False = cisto retka nagrada (0 dok se ne zavrsi, onda max).
        (ostali parametri identicni standardnim robosuite environmentima)
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
        assert source_peg_idx != target_peg_idx, "source_peg_idx i target_peg_idx moraju biti razliciti"
        assert source_peg_idx in (0, 1, 2) and target_peg_idx in (0, 1, 2), "indeksi klinova su 0, 1 ili 2"

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        # tri logicke pozicije klinova, raspoređene levo-desno (duz y ose)
        # relativno u odnosu na centar stola
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

        # polovine ivica kocki (m) -- opadajuce velicine, najveca ide na dno
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

    # ------------------------------------------------------------------
    # Pomocne geometrijske / provere stanja
    # ------------------------------------------------------------------

    def _peg_world_xy(self, peg_idx):
        """Apsolutna (x, y) pozicija klina @peg_idx u mujoco world koordinatama."""
        return self.table_offset[:2] + self.peg_offsets[peg_idx]

    def _cube_at_peg(self, cube, peg_idx, xy_tol=0.03):
        """True ako je (x, y) pozicija kocke unutar tolerancije od klina."""
        pos_xy = np.array(self.sim.data.body_xpos[self.obj_body_id[cube.name]])[:2]
        return np.linalg.norm(pos_xy - self._peg_world_xy(peg_idx)) < xy_tol

    def _b_on_a(self):
        """True ako je cubeB fizicki na cubeA i robot je vise ne drzi."""
        grasping_b = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeB)
        return (not grasping_b) and self.check_contact(self.cubeB, self.cubeA)

    def _c_on_b(self):
        """True ako je cubeC fizicki na cubeB i robot je vise ne drzi."""
        grasping_c = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)
        return (not grasping_c) and self.check_contact(self.cubeC, self.cubeB)

    # ------------------------------------------------------------------
    # Nagrada
    # ------------------------------------------------------------------

    def reward(self, action):
        """
        Delimicni bonus (partial credit), koristan i za sparse i za shaped rezim:

            +1.0  cubeA (najveca, donja) je stigla na ciljni klin
            +1.0  cubeB je na cubeA (znaci: kula se ponovo gradi na cilju)
            +2.0  cubeC je na cubeB (kula kompletna -> uspeh)

        Namerno NE prati posebno svaki od 7 optimalnih poteza -- za
        snimanje ljudskih demonstracija to nije potrebno; operater sam
        resava kako da oslobodi cubeA (mora prvo da skloni B i C, verovatno
        na treci/pomocni klin).

        Ako je reward_shaping=False (podrazumevano), nagrada je cisto
        sparse: 0 dok zadatak nije potpuno zavrsen, max iznos na kraju.
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

    # ------------------------------------------------------------------
    # Izgradnja scene
    # ------------------------------------------------------------------

    def _load_model(self):
        """Sto, robot i tri kocke (bez placement_initializer-a -- pozicije
        se rucno postavljaju u _reset_internal jer kula mora biti VEC
        slozena na pocetku, sto standardni random sampler ne podrzava)."""
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

        # Vizuelni markeri za tri klina -- tanki, ravni diskovi bez kolizije
        # (joints=None znaci "staticno telo", obj_type="visual" znaci "samo
        # za prikaz, ne ucestvuje u fizici"). Pozicija se postavlja RUCNO na
        # XML elementu PRE spajanja u scenu (get_obj().set("pos", ...)) jer
        # staticni objekti nemaju joint preko kog bi se pozicija menjala u
        # _reset_internal, kao sto to radimo za kocke.
        #
        # color_code_pegs=True boji izvorni/ciljni klin razlicito -- KORISNO
        # ZA TVOJE SOPSTVENO TESTIRANJE, ali namerno OFF po difoltu: VLA
        # model treba da uci "koji je cilj" iz JEZICKE instrukcije, ne iz
        # boje koja ne postoji na pravom stolu van simulacije. Iskljuci ovo
        # (podrazumevano vrednost) kad snimas prave demonstracije za dataset.
        #
        # NAPOMENA: boje se postavljaju OVDE, JEDNOM, na osnovu vrednosti
        # source_peg_idx/target_peg_idx iz konstruktora -- ako kasnije
        # koristis randomize_pegs=True, boje ce odgovarati samo PRVOM
        # (konstrukcionom) izboru, ne ce se azurirati na svaki reset. Za
        # taj slucaj, oslanjaj se na konzolni ispis izvor/cilj klina umesto
        # na boju.
        self.peg_markers = []
        marker_z = self.table_offset[2] + 0.001  # tik iznad povrsine stola
        for idx, offset in self.peg_offsets.items():
            if self.color_code_pegs and idx == self.source_peg_idx:
                rgba = [0.9, 0.55, 0.1, 0.8]   # narandzasto = IZVOR
            elif self.color_code_pegs and idx == self.target_peg_idx:
                rgba = [0.15, 0.75, 0.15, 0.8]  # zeleno = CILJ
            else:
                rgba = [0.25, 0.25, 0.25, 0.6]  # neutralno sivo (podrazumevano, ili pomocni klin)

            marker = CylinderObject(
                name=f"peg{idx}_marker",
                size=[0.045, 0.001],  # radijus, polovina visine -- vrlo tanak disk
                rgba=rgba,
                joints=None,
                obj_type="visual",
            )
            marker_pos = np.array(
                [self.table_offset[0] + offset[0], self.table_offset[1] + offset[1], marker_z]
            )
            marker.get_obj().set("pos", array_to_string(marker_pos))
            self.peg_markers.append(marker)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes + self.peg_markers,
        )

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {
            "cubeA": self.sim.model.body_name2id(self.cubeA.root_body),
            "cubeB": self.sim.model.body_name2id(self.cubeB.root_body),
            "cubeC": self.sim.model.body_name2id(self.cubeC.root_body),
        }

    def _reset_internal(self):
        """Postavlja kompletnu, VEC slozenu kulu na izvorni klin (source_peg_idx).
        Ovo je kljucna razlika u odnosu na standardni robosuite placement_initializer,
        koji nezavisno randomizuje svaki objekat -- ovde nam treba da sve tri
        kocke pocnu poravnate/slozene jedna na drugoj."""
        super()._reset_internal()

        if not self.deterministic_reset:
            if self.randomize_pegs:
                src, tgt = self.rng.choice(3, size=2, replace=False)
                self.source_peg_idx, self.target_peg_idx = int(src), int(tgt)
            # ako randomize_pegs=False, ostaju vrednosti iz konstruktora

            print(f"[HanoiThree] izvorni klin: {self.source_peg_idx}, ciljni klin: {self.target_peg_idx}")

            # sitan xy jitter (isti za sve tri kocke, da ostanu poravnate) --
            # sprecava da model nauci fiksnu, uvek identicnu pocetnu pozu
            xy = self._peg_world_xy(self.source_peg_idx) + self.rng.uniform(-0.005, 0.005, size=2)

            identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z), bez rotacije
            z_cursor = self.table_offset[2]  # povrsina stola

            for name in ["cubeA", "cubeB", "cubeC"]:
                half_h = self.half_heights[name]
                z_center = z_cursor + half_h
                obj = getattr(self, name)
                pos = np.array([xy[0], xy[1], z_center])
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, identity_quat]))
                z_cursor = z_center + half_h  # vrh ove kocke = dno sledece

    def _setup_observables(self):
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

            # pozicija ciljnog klina -- korisno kao dodatni signal za VLA model
            # (npr. "gde treba da zavrsi kula", posto se cilj menja ako
            # randomize_pegs=True)
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

    # ------------------------------------------------------------------
    # Uspeh i vizuelizacija
    # ------------------------------------------------------------------

    def _check_success(self):
        """Kula je uspesno premestena kad je cubeA na CILJNOM klinu, sa
        cubeB na cubeA i cubeC na cubeB -- tj. cela struktura je ponovo
        sastavljena, ali na drugom mestu nego na pocetku."""
        return (
            self._cube_at_peg(self.cubeA, self.target_peg_idx)
            and self._b_on_a()
            and self._c_on_b()
        )

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            # cubeC je (skoro) uvek prvi na potezu jer je na vrhu pocetne kule
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeC)


# ==========================================================================
# Primer pokretanja / brzi sanity-check
# ==========================================================================
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
    print(f"Izvorni klin: {env.source_peg_idx}, ciljni klin: {env.target_peg_idx}")

    for step in range(100):
        action = np.zeros(env.action_dim)  # placeholder -- ovde ide tvoja IK akcija
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(1.0 / env.control_freq)
        if step % 20 == 0:
            print(f"step {step}: reward={reward:.3f}, success={env._check_success()}")
        if done:
            obs = env.reset()

    input("Pritisni Enter da zatvoris...")
    env.close()