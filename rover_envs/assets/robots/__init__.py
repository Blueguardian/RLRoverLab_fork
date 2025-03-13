from rover_envs.assets.utils.Gen.ConfigFactory import ConfigFactory, GymEnvRegistrar
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg
from pathlib import Path

ROOT_FOLDER = Path(__file__).parent

CONFIG_FACTORY = ConfigFactory(ROOT_FOLDER, RoverEnvCfg)
DYNAMIC_ENV_CLASSES = CONFIG_FACTORY.get_cfgclasses()
GymEnvRegistrar(CONFIG_FACTORY)

globals().update(DYNAMIC_ENV_CLASSES)


