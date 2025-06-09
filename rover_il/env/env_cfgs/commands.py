from __future__ import annotations

from typing import TYPE_CHECKING

import os
import math
from isaaclab.utils import configclass
from rover_envs.assets.terrains.debug.debug_terrains import DebugTerrainSceneCfg  # noqa: F401
from rover_envs.assets.terrains.mars import MarsTerrainSceneCfg  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.commands_cfg import TerrainBasedPositionCommandCfg  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommand  # noqa: F401


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_pose = TerrainBasedPositionCommandCfg(
        class_type=TerrainBasedPositionCommand,  # TerrainBasedPositionCommandCustom,
        asset_name="robot",
        rel_standing_envs=0.0,
        simple_heading=False,
        resampling_time_range=(150.0, 150.0),
        ranges=TerrainBasedPositionCommandCfg.Ranges(
            heading=(-math.pi, math.pi)),
        debug_vis=True,
    )