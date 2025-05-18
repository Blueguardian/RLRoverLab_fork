from __future__ import annotations

import inspect # noqa: F401
import gymnasium as gym
import ruamel.yaml # noqa: F401
from pathlib import Path # noqa: F401
from prettytable import PrettyTable

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from rover_envs.mdp.actions.actions_cfg import ACTION_CONFIGS, SkidSteeringSimpleCfg
# from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg
from rover_envs.envs.navigation.learning.skrl import get_agent
from rover_envs.envs.navigation.utils.articulation.articulation import RoverArticulation

yaml = ruamel.yaml.YAML(typ="safe")

# Global dict to store configs
CONFIG_CLASSES = {}

def post_init_gen(robot_config, articulation_cfg, action_cfg):
    """Factory function that generates a `__post_init__` method for dynamic classes."""
    def __post_init__(self):
        super(self.__class__, self).__post_init__()
        self.scene.robot = articulation_cfg
        self.actions.actions = action_cfg

    return __post_init__

class configGen:
    """ Dynamically allocate and initialize configuration classes for
        each new asset. Assets needs to have config files in a "configs"
        folder.

        Utilization: Initializes through importing the robots folder anywhere
        :Watchdog: The watchdog should be active if a new asset is added as it
        will provide the default configuration files and folder setup
        """

    def __init__(self, root_folder: Path, base_class):

        # The target folder (asset/robots) -> Could be hard-coded
        self.root_folder = root_folder

        # The class which each class should inherit from -> Could be hard-coded
        self.parent_class = configclass(base_class)
        print("Generating asset configurations...") # Just info prompt

        # Setup for printing tables
        self._table = PrettyTable()
        self._table.field_names = ["Config Class", "Asset Folder"]
        self._table.align = "l"

        # Generate classes
        self.config_classes = self._generate_cfgs()

        # Finalize by printing the configs
        print("\n+------------------------------+")
        print("|     Generated Configs        |")
        print("+------------------------------+")
        print(self._table)


    def _load_yaml(self, file_path: Path):
        """ Loads yaml files
            Used mainly for opening
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Missing configuration file: {file_path}")
        with open(file_path, "r") as f:
            return yaml.load(f)

    def _class_name(self, folder: Path, params: dict):
        """ Generate class name based on folder name or from config
            Allows a definition of a custom class name if specified in config

            Not added to config as standard
            """
        folder = folder.name
        # Either use the config or use the folder name
        return params.get("class_config", {}).get("config_name") or \
               "".join(word.capitalize() for word in folder.replace("-", "_").split("_")) + "EnvCfg"

    def _nestedDict(self, cfg_class, cfg_dict):
        """ Handles nested dicts in config file

            Due to the setup of the configuration file
            it is required to handle nested dicts
            could perhaps be changed depending on the setup
            of the config file"""
        if not cfg_dict:
            return None
        # Check the parameters of the cfg class
        cfg_params = inspect.signature(cfg_class).parameters

        # Filter the inputs according to the required parameters and default values
        filtered_params = {
            param: cfg_dict.get(param,
                                   param_info.default if param_info.default is not inspect.Parameter.empty else None)
            for param, param_info in cfg_params.items()
        }
        # Return the class with the given parameters
        return cfg_class(**filtered_params)

    def _joint_values(self, joint_cfg, key):
        """Extracts joint values from config file since they are nested"""
        if not joint_cfg or key not in joint_cfg:
            return {}
        names = joint_cfg[key].get("names", [])
        values = joint_cfg[key].get("values", [])
        return {name: values[i] if i < len(values) else 0.0 for i, name in enumerate(names)}

    def _pose_config(self, params):
        """Handles pose initialization"""
        pose_cfg = params.get("pose_config", {})
        return ArticulationCfg.InitialStateCfg(
            pos=tuple(pose_cfg.get("init_pose", (0.0, 0.0, 0.0))),
            joint_pos=self._joint_values(pose_cfg.get("joint_pose", {}), "pose_config"),
            joint_vel=self._joint_values(pose_cfg.get("joint_pose", {}), "velocity_config"),
        )
    def _actuators(self, params):
        """Extracts actuator configurations"""
        return {
            actuator: ImplicitActuatorCfg(**actuator_data)
            for actuator, actuator_data in params.get("joints", {}).items()
        }
    def _action_cfg(self, params):
        """Selects and instantiates the action controller"""
        action_type = str(params.get("controller_config"))[:-1]
        ActionConfigClass = ACTION_CONFIGS.get(action_type, SkidSteeringSimpleCfg)
        action_params = inspect.signature(ActionConfigClass).parameters
        filtered_params = {
            param: params["controller"].get(param, param_info.default if param_info.default is not inspect.Parameter.empty else None)
            for param, param_info in action_params.items()
        }
        filtered_params.setdefault("asset_name", "robot")
        return ActionConfigClass(**filtered_params)

    def _articulation_cfg(self, params):
        """Handles the ArticulationCfg instance and instantiates it with filtered parameters"""
        usd_path = Path(__file__).resolve().parent.parent.parent / ("robots" + str(params.get("robot_model_path")).split(",")[0])


        ArticulationRobotCfg =  ArticulationCfg(
            class_type=RoverArticulation,
            spawn=self._nestedDict(sim_utils.UsdFileCfg, {
                "usd_path": usd_path.absolute().as_posix(),
                "activate_contact_sensors": True,
                "collision_props": self._nestedDict(sim_utils.CollisionPropertiesCfg, params.get("collision_properties")),
                "rigid_props": self._nestedDict(sim_utils.RigidBodyPropertiesCfg, params.get("rigidBody_properties")),
                "articulation_props": self._nestedDict(sim_utils.ArticulationRootPropertiesCfg, params.get("simulation_properties")),
            }),
            init_state=self._pose_config(params),
            actuators=self._actuators(params),
            )
        return ArticulationRobotCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def _generate_cfgs(self):
        """Creates configs based on each folder in the root_folder"""
        env_cfgs = {}

        for folder in self.root_folder.iterdir():
            if not folder.is_dir() or folder.name == "__pycache__" or folder.name == ".vscode":
                continue

            # Load YAML configs
            robot_cfg = self._load_yaml(folder / "configs" / "robot_default.yaml")
            training_cfg = self._load_yaml(folder / "configs" / "training_default.yaml")
            # Generate or fetch class name
            class_name = self._class_name(folder, robot_cfg)

            # Define and register config class
            attributes = {
                "__doc__": f"Configuration for {folder.name} rover environement.",
                "__post_init__": post_init_gen(robot_cfg, self._articulation_cfg(robot_cfg), self._action_cfg(robot_cfg))
            }

            # Create the class: TODO: remove __class__
            cls = type(class_name, (self.parent_class, self.__class__), attributes)
            cls = configclass(cls)
            env_cfgs[class_name] = cls

            #Testing
            self._table.add_row([class_name, folder.name])
        return env_cfgs

    def get_cfgclasses(self):
        """Returns the generated configuration classes"""
        return self.config_classes


class GymEnvRegistrar:
    """
    Dynamically registers Gymnasium-compatible environments based on available robot asset configurations.

    This class parses YAML configuration files inside each asset's `configs/` folder, extracts
    relevant training and task parameters (such as RL algorithms, task name, and agent configuration paths),
    and registers the environment with the `gymnasium` registry using a unique `env_id`.

    The configuration classes required for each environment are dynamically created using
    `configGen`, which must be passed in as a dependency.

    Each registered environment becomes accessible through standard Gym APIs,
    and includes metadata such as:
        - A custom configuration class for the simulation setup
        - Best policy model path
        - Agent setup function (`get_agent`)
        - Configuration paths for supported algorithms

    Example Gym ID: "aau_rover-v0"

    Output:
        - A formatted table summarizing the registered environments, printed to the console.

    Args:
        config_factory (configGen): Instance that provides dynamically generated environment config classes.

    Attributes:
        config_factory (configGen): Source of environment configuration classes.
        base_dir (Path): Base path of the file for relative path resolution.
        _table (PrettyTable): Table for summarizing environment registration status.
    """

    def __init__(self, config_factory: configGen):
        self.config_factory = config_factory  # Use the generated config classes
        self.base_dir = Path(__file__).parent
        print("Registering gym environments...")

        self._table = PrettyTable()
        self._table.field_names = ["Gym ID", "Asset Folder", "Algorithms"]
        self._table.align = "l"

        self.register_envs()

        print("\n+------------------------------+")
        print("|     Registered Environments  |")
        print("+------------------------------+")
        print(self._table)

    def _load_yaml(self, file_path: Path):
        """Loads a YAML configuration file"""
        if not file_path.is_file():
            return None
        with file_path.open("r") as stream:
            return yaml.load(stream)

    def register_envs(self):
        """
    Iterates through all asset folders and registers corresponding Gym environments.

    - Loads each environment's training configuration from `training_default.yaml`.
    - Extracts the list of supported RL algorithms.
    - Constructs paths to the SKRL agent configs.
    - Matches the config class by generating a name from the folder structure.
    - Registers the environment with `gymnasium`.

    If no configuration or config class is found, the environment is skipped.
    """
        for env_folder in self.config_factory.root_folder.iterdir():
            if not env_folder.is_dir():
                continue

            #Load per-environment configuration
            learning_config = self._load_yaml(env_folder / "configs" / "training_default.yaml")
            if not learning_config:
                continue

            #Extract allowed RL algorithms
            algorithms = learning_config.get("algorithms", ["PPO"])

            #Generate paths for SKRL agent configs
            skrl_configs = {
                algo: str(Path(__file__).parent.parent.parent.parent / f"envs/navigation/learning/skrl/configs/rover_{algo.lower()}.yaml")
                for algo in algorithms
            }

            #Get the dynamically generated env config class using folder-based naming
            class_name = "".join(word.capitalize() for word in env_folder.name.replace("-", "_").split("_")) + "EnvCfg"
            env_config_class = self.config_factory.get_cfgclasses().get(class_name, None)

            if not env_config_class:
                continue

            # Generate Gym ID based on folder name or config definition
            env_id = f"{env_folder.name}-v0" if not "task_name" in learning_config else learning_config.get("task_name")+"-v0"

            #Register the environment in Gym
            gym.register(
                id=env_id,
                entry_point='rover_envs.envs.navigation.entrypoints:RoverEnv',
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": env_config_class,
                    "best_model_path": Path(env_folder, "policies/best_agent.pt").absolute().as_posix(),
                    "get_agent_fn": get_agent,
                    "skrl_cfgs": skrl_configs,
                },
            )
            self._table.add_row([env_id, env_folder.name, ", ".join(algorithms)])

