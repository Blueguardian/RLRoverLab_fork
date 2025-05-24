from __future__ import annotations

from typing import TYPE_CHECKING

import math
from isaaclab.scene import InteractiveSceneCfg  # noqa: F401
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs.mdp import *
from isaaclab.envs.mdp import last_action
from isaaclab.utils.noise import NoiseCfg, AdditiveGaussianNoiseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, mdp




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
def height_scan_rover(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculate the height scan of the rover.

    This function uses a ray caster to generate a height scan of the rover's surroundings.
    The height scan is normalized by the maximum range of the ray caster.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - 0.26878
    # Note: 0.26878 is the distance between the sensor and the rover's base
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878
def relative_orientation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle difference between the rover's heading and the target."""
    # Get the angle to the target
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]
    return heading_angle_diff.unsqueeze(-1)
def image(env, sensor_cfg, data_type):
    sensor = env.scene.sensors.get(sensor_cfg.name)
    output = sensor.data.output.get(data_type)
    if data_type == "rgb":
        try:
            if not isinstance(output, torch.Tensor):
                output = torch.from_numpy(output)
            output = output.to(torch.float32)
            output = output.permute(0, 3, 1, 2)  # BCHW
            return output
        except Exception as e:
            print(f"[ERROR RGB] Failed to process RGB output: {e}")
            return torch.zeros((1, 3, 100, 100), dtype=torch.float32, device=env.device)
    elif data_type == "depth":
        try:
            depth = sensor.data.output["depth"]
            depth = depth.permute(0, 3, 1, 2)
            return depth
        except Exception as e:
            print(f"[ERROR] Failed to handle depth output: {e}")
            return torch.zeros((1, 1, 100, 100), dtype=torch.float32, device=env.device)
    elif data_type == "RaycasterCamera":
        try:
            depth = sensor.data.output["distance_to_image_plane"]
            depth = depth.permute(0, 3, 1, 2)
            return depth
        except Exception as e:
            print(f"[ERROR] Failed to handle depth output: {e}")
            return torch.zeros((1, 1, 100, 100), dtype=torch.float32, device=env.device)
    else:
        print(f"[ERROR] Unknown data_type: {data_type}")
        return torch.zeros((1, 3, 100, 100), dtype=torch.float32, device=env.device)

from isaaclab.utils.noise import AdditiveGaussianNoiseCfg
from rover_il.env.noise_cfgs.redwood import RedwoodDepthNoiseCfg
from rover_il.env.noise_cfgs.gaussian import GaussianImageNoiseCfg

@configclass
class ObservationCfg:
    """Observation configuration for the task."""

    @configclass
    class PolicyCfg(ObsGroup):

        # Teacher terms
        actions_teacher = ObsTerm(
            func=last_action
        )
        distance_teacher = ObsTerm(
            func=distance_to_target_euclidean,
            params={
                "command_name": "target_pose"
            },
            scale=0.11
        )
        heading_teacher = ObsTerm(
            func=angle_to_target_observation,
            params={
                "command_name": "target_pose",
            },
            scale=1 / math.pi,
        )
        relative_goal_orientation_teacher = ObsTerm(
            func=relative_orientation,
            params={
                "command_name": "target_pose"
            },
            scale=1 / math.pi
        )
        height_scan_teacher = ObsTerm(
            func=height_scan_rover,
            params={
                "sensor_cfg": SceneEntityCfg(name="height_scanner")
            },
            scale=1,
        )

        # Student observations
        actions_student = ObsTerm(
            func=last_action,
        )
        distance_student = ObsTerm(
            func=distance_to_target_euclidean,
            params={
            "command_name": "target_pose"
            },
            scale=0.11
        )
        heading_student = ObsTerm(
            func=angle_to_target_observation,
            params={
                "command_name": "target_pose",
            },
            scale=1 / math.pi,
        )
        relative_goal_orientation_student = ObsTerm(
            func=relative_orientation,
            params={
                "command_name": "target_pose"
            },
            scale=1 / math.pi
        )
        camera_rgb_student = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
            },
            noise=AdditiveGaussianNoiseCfg,
        )
        camera_depth_student = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "depth",
            }
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @classmethod
    def from_yaml_dict(cls, cfg: dict, agent: str) -> ObservationCfg:

        NOISE_MODELS = {
            "gaussian": lambda: GaussianImageNoiseCfg(),
            "redwood": lambda: RedwoodDepthNoiseCfg(),
        }

        full_policy = cls.PolicyCfg()
        filtered_terms = {}

        for key, val in vars(full_policy).items():
            if not isinstance(val, ObsTerm):
                continue
            if not key.endswith(f"_{agent}"):
                continue

            base_key = key.removesuffix(f"_{agent}")
            agent_cfg = cfg.get(base_key, {}).get(agent, {})

            if not agent_cfg.get("enabled", False):
                continue

            term = val.replace()

            if "scale" in agent_cfg:
                term = term.replace(scale=agent_cfg["scale"])

            if "noise" in agent_cfg:
                noise_type = agent_cfg["noise"]
                if noise_type in NOISE_MODELS:
                    term = term.replace(noise=NOISE_MODELS[noise_type]())
                else:
                    raise ValueError(f"Unknown noise type '{noise_type}' in observation term '{key}'")

            filtered_terms[key] = term

        policy_cfg = cls.PolicyCfg(**filtered_terms)
        obs_cfg = cls()
        obs_cfg.policy = policy_cfg
        return obs_cfg

    policy: PolicyCfg = PolicyCfg()