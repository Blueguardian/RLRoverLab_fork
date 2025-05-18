from pathlib import Path
import inspect
import ruamel.yaml
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, CollisionPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.actuators import ImplicitActuatorCfg
from rover_il.agent.agent_cfg.actions import ActionsCfg

from rover_il.env.env_cfgs.observations import ObservationCfg
from rover_il.env.env_cfgs.rewards import RewardsCfg
from rover_il.env.env_cfgs.terminations import TerminationsCfg
from rover_il.env.env_cfgs.commands import CommandsCfg
from rover_il.env.env_cfgs.curriculums import CurriculumCfg
from rover_il.env.env_cfgs.events import EventCfg
from rover_il.env.terrain_cfg.scenecfg import RoverSceneCfg
from rover_envs.mdp.actions.actions_cfg import ACTION_CONFIGS, SkidSteeringSimpleCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg
from rover_envs.envs.navigation.utils.articulation.articulation import RoverArticulation

import gymnasium as gym

yaml = ruamel.yaml.YAML(typ="safe")


class ILEnvironmentConfigurator:
    def __init__(self, root_path: Path, config_root: Path):
        # root_path should point to monorepo root: /workspace/isaac_rover
        self.root_path = root_path.resolve()
        self.config_root = config_root.resolve()

    def _load_yaml(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            return yaml.load(f)

    def _resolve_usd_path(self, rel_path: str) -> str:
        rel_path_clean = rel_path.lstrip("/")
        base_path = self.root_path / "rover_envs/assets/robots"
        full_path = (base_path / rel_path_clean).resolve()
        if not full_path.exists():
            raise FileNotFoundError(f"[USD Resolver] Missing: {full_path}")
        return full_path.as_posix()

    def _extract_joint_dict(self, cfg: dict, key: str) -> dict:
        names = cfg.get("pose_config", {}).get("joint_pose", {}).get(key, {}).get("names", [])
        values = cfg.get("pose_config", {}).get("joint_pose", {}).get(key, {}).get("values", [])
        return {name: values[i] if i < len(values) else 0.0 for i, name in enumerate(names)}

    def _nestedDict(self, cfg_class, cfg_dict):
        if not cfg_dict:
            return cfg_class()
        cfg_params = inspect.signature(cfg_class).parameters
        filtered_params = {
            param: cfg_dict.get(param, param_info.default if param_info.default is not inspect.Parameter.empty else None)
            for param, param_info in cfg_params.items()
        }
        return cfg_class(**filtered_params)

    def _action_cfg(self, params: dict):
        ctrl_key = str(params.get("controller_config", "")).lower().rstrip(":")
        ActionCfgCls = ACTION_CONFIGS.get(ctrl_key, SkidSteeringSimpleCfg)
        action_params = inspect.signature(ActionCfgCls).parameters
        filtered_params = {
            param: params["controller"].get(param, param_info.default if param_info.default is not inspect.Parameter.empty else None)
            for param, param_info in action_params.items()
        }
        filtered_params.setdefault("asset_name", "robot")
        return ActionCfgCls(**filtered_params)

    def create_env_cfg(self, config_dir: Path, role: str):
        agent_cfg = self._load_yaml(config_dir / "agent_config.yaml")
        env_cfg = self._load_yaml(config_dir / "environment_config.yaml")
        train_cfg = self._load_yaml(config_dir / "learning_config.yaml")

        actuators = {
            name: ImplicitActuatorCfg(**cfg)
            for name, cfg in agent_cfg.get("joints", {}).items()
        }
        usd_path = self._resolve_usd_path(agent_cfg["robot_model_path"])

        articulation = ArticulationCfg(
            class_type=RoverArticulation,
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=True,
                collision_props=CollisionPropertiesCfg(**agent_cfg.get("collision_properties", {})),
                rigid_props=RigidBodyPropertiesCfg(**agent_cfg.get("rigidBody_properties", {})),
                articulation_props=self._nestedDict(ArticulationRootPropertiesCfg, agent_cfg.get("simulation_properties", {})),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(agent_cfg.get("pose_config", {}).get("init_pose", [0.0, 0.0, 0.0])),
                joint_pos=self._extract_joint_dict(agent_cfg, "pose_config"),
                joint_vel=self._extract_joint_dict(agent_cfg, "velocity_config"),
            ),
            actuators=actuators,
        )

        action_cfg = self._action_cfg(agent_cfg)
        sim_cfg = SimulationCfg(dt=train_cfg["simulation"]["dt"], physx=PhysxCfg(**env_cfg["physx"]))
        scene_cfg = RoverSceneCfg(num_envs=128, env_spacing=4.0, replicate_physics=False, robot=articulation)
        obs_cfg = ObservationCfg.from_yaml_dict(train_cfg["observations"], agent=role)
        rew_cfg = RewardsCfg.from_yaml_dict(train_cfg["rewards"])
        term_cfg = TerminationsCfg.from_yaml_dict(train_cfg["terminations"])

        class_name = f"Rover{role.capitalize()}EnvCfg"

        def __post_init__(self):
            self.sim.dt = train_cfg["simulation"]["dt"]
            self.decimation = train_cfg["simulation"]["decimation"]
            self.episode_length_s = train_cfg["simulation"]["episode_length_s"]
            self.scene.robot = articulation
            self.actions.actions = action_cfg
            self.observations = obs_cfg
            self.rewards = rew_cfg
            self.terminations = term_cfg
            self.commands = CommandsCfg()
            self.curriculum = CurriculumCfg()
            self.events = EventCfg()
            self.viewer.eye = (-6.0, -6.0, 3.5)

        cfg_class = type(class_name, (ManagerBasedRLEnvCfg,), {"__post_init__": __post_init__})
        return configclass(cfg_class)(
            sim=sim_cfg,
            scene=scene_cfg,
            actions=ActionsCfg(actions={}),
            decimation=train_cfg["simulation"]["decimation"],
            episode_length_s=train_cfg["simulation"]["episode_length_s"],
        )

    def generate_all_cfg_classes(self) -> dict:
        cfg_classes = {}
        for robot_dir in self.config_root.iterdir():
            config_dir = robot_dir / "configs"
            if not config_dir.is_dir():
                continue

            train_cfg = self._load_yaml(config_dir / "learning_config.yaml")
            agent_cfg = self._load_yaml(config_dir / "agent_config.yaml")

            modes = train_cfg.get("modes", ["DAgger"])
            task_name = train_cfg.get("task_name", robot_dir.name)
            version = agent_cfg.get("version", 0)

            for mode in modes:
                roles = ["teacher", "student"] if mode == "DAgger" else ["expert"]
                for role in roles:
                    class_name = f"{task_name.capitalize()}{role.capitalize()}EnvCfg"
                    cfg_class = self.create_env_cfg(config_dir, role)
                    cfg_classes[class_name] = cfg_class
        return cfg_classes

    def register_envs(self):
        for robot_dir in self.config_root.iterdir():
            config_dir = robot_dir / "configs"
            if not config_dir.is_dir():
                continue

            train_cfg = self._load_yaml(config_dir / "learning_config.yaml")
            agent_cfg = self._load_yaml(config_dir / "agent_config.yaml")

            modes = train_cfg.get("modes", ["DAgger"])
            task_name = train_cfg.get("task_name", robot_dir.name)
            version = agent_cfg.get("version", 0)
            base_env_id = f"{task_name}-v{version}"

            for mode in modes:
                roles = ["teacher", "student"] if mode == "DAgger" else ["expert"]
                for role in roles:
                    env_cfg = self.create_env_cfg(config_dir, role)
                    skrl_cfgs = {
                        algo: str(self.root_path / f"rover_envs/envs/navigation/learning/skrl/configs/rover_{algo.lower()}.yaml")
                        for algo in train_cfg.get("algorithms", ["PPO"])
                    }
                    env_id = base_env_id if role == "student" else f"{base_env_id}-{role}"
                    gym.register(
                        id=env_id,
                        entry_point='rover_envs.envs.navigation.entrypoints:RoverEnv',
                        disable_env_checker=True,
                        kwargs={"env_cfg_entry_point": env_cfg, "skrl_cfgs": skrl_cfgs},
                    )
