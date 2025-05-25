from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import MISSING

import os
import rover_envs
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from rover_envs.assets.terrains.debug.debug_terrains import DebugTerrainSceneCfg  # noqa: F401
from rover_envs.assets.terrains.mars import MarsTerrainSceneCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg, TiledCamera, Camera, CameraCfg, RayCasterCamera, RayCasterCameraCfg



@configclass
class RoverSceneCfg(MarsTerrainSceneCfg):
    """
    Rover Scene Configuration

    Note:
        Terrains can be changed by changing the parent class e.g.
        RoverSceneCfg(MarsTerrainSceneCfg) -> RoverSceneCfg(DebugTerrainSceneCfg)

    """

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color_temperature=4500.0,
            intensity=100,
            enable_color_temperature=True,
            texture_file=os.path.join(
                os.path.dirname(os.path.abspath(rover_envs.__path__[0])),
                "rover_envs",
                "assets",
                "textures",
                "background.png",
            ),
            texture_format="latlong",
        ),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=30000.0, radius=50, color_temperature=5500, enable_color_temperature=True
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -180.0, 80.0)),
    )

    robot: ArticulationCfg = MISSING

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(Drive|Steer|Boogie|Body|Rocker|Drive_Link)",
        filter_prim_paths_expr=["/World/terrain/obstacles/obstacles"],
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Main_Body",
        offset=RayCasterCfg.OffsetCfg(pos=[0.0, 0.0, 10.0]),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[5.0, 5.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/terrain/hidden_terrain"],
        max_distance=100.0,
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/.*/Main_Body/Camera1",
        depth_clipping_behavior='max',
        data_types=["rgb", "depth"],
        width=100, height=100,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.00, focus_distance=400.0,
            horizontal_aperture=32.000, clipping_range=(0.1, 10.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.55, 0.0, 0.6),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="parent"
        )
    )