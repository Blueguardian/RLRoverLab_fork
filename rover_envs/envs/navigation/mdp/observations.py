from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, Camera, CameraCfg

# from isaaclab.command_generators import UniformPoseCommandGenerator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def angle_to_target_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle to the target."""

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])
    return angle.unsqueeze(-1)


def distance_to_target_euclidean(env: ManagerBasedRLEnv, command_name: str):
    """Calculate the euclidean distance to the target."""
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    distance: torch.Tensor = torch.norm(target_position, p=2, dim=-1)
    return distance.unsqueeze(-1)


# def height_scan_rover(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Calculate the height scan of the rover.
#
#     This function uses a ray caster to generate a height scan of the rover's surroundings.
#     The height scan is normalized by the maximum range of the ray caster.
#     """
#     # extract the used quantities (to enable type-hinting)
#     sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
#     # height scan: height = sensor_height - hit_point_z - 0.26878
#     # Note: 0.26878 is the distance between the sensor and the rover's base
#     return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878


def angle_diff(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle difference between the rover's heading and the target."""
    # Get the angle to the target
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]
    return heading_angle_diff.unsqueeze(-1)

def image(env, sensor_cfg, data_type):
    sensor = env.scene.sensors.get(sensor_cfg.name)
    output = sensor.data.output.get(data_type)

    if data_type == "rgb":
        try:
            if isinstance(output, torch.Tensor):
                pass  # already a tensor
            else:
                output = torch.from_numpy(output)

            if output.dtype == torch.uint8:
                output = output.float() / 255.0
            output = output.permute(0, 3, 1, 2)
            return output
        except Exception as e:
            print(f"[ERROR RGB] Failed to process RGB output: {e}")
            return torch.zeros((1, 3, 112, 112), dtype=torch.float32, device=env.device)
    elif data_type == "depth":
        try:
            depth = sensor.data.output.get("depth")
            if torch.isinf(depth).any():
                depth = torch.nan_to_num(depth, nan=0.0, posinf=10.0, neginf=0.0)
            depth = depth.permute(0, 3, 1, 2)  # to [B, 1, H, W]
            return depth
        except Exception as e:
            print(f"[ERROR] Failed to handle depth output: {e}")
            return torch.zeros((1, 224, 224), dtype=torch.float32)

    else:
        print(f"[ERROR] Unknown data_type: {data_type}")
        return torch.zeros((1, 3, 112, 112), dtype=torch.float32, device=env.device)