from rover_envs.assets.utils.asset_gen.config_gen import configGen, GymEnvRegistrar
from rover_envs.assets.utils.env_gen.env_gen import RoverEnvCfg
from pathlib import Path

# Assets/robots folder
ROOT_FOLDER = Path(__file__).parent

# Initialize all confguration files
CONFIG_FACTORY = configGen(ROOT_FOLDER, RoverEnvCfg)
ENV_CLASSES = CONFIG_FACTORY.get_cfgclasses()
# Register all in gymnasium
GymEnvRegistrar(CONFIG_FACTORY)

# Update the globals for importing classes
globals().update(ENV_CLASSES)


