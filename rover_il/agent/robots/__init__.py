from pathlib import Path
from rover_il.utils.il_config_gen import ILEnvironmentConfigurator

# Go up 1 extra level: from /rover_il/agent/robots to /workspace/isaac_rover
ROOT_PATH = Path(__file__).resolve().parents[3]  # ✅ monorepo root
CONFIG_ROOT = Path(__file__).resolve().parent   # /rover_il/agent/robots

CONFIG_FACTORY = ILEnvironmentConfigurator(ROOT_PATH, CONFIG_ROOT)
CONFIG_FACTORY.register_envs()

ENV_CLASSES = CONFIG_FACTORY.generate_all_cfg_classes()
globals().update(ENV_CLASSES)