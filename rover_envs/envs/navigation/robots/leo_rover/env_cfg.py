from __future__ import annotations

from isaaclab.utils import configclass

import rover_envs.mdp as mdp
from rover_envs.assets.robots.leo_rover import LEO_ROVER_CFG
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg


@configclass
class LeoRoverEnvCfg(RoverEnvCfg):
    """Configuration for the Leo rover environment."""

    def __post_init__(self):
        super().__post_init__()

        # Define robot
        self.scene.robot = LEO_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Define parameters for the Skidsteering kinematics.
        self.actions.actions = mdp.SkidSteeringSimpleCfg(
            asset_name="robot",
            wheelbase_length=0.37,
            wheel_radius=0.0635,
            drive_joint_names=[".*Drive_Joint"],
            slip_factor=0
            # offset=-0.0135
        )
