from __future__ import annotations
from typing import TYPE_CHECKING

import torch
# Importing necessary modules from the isaaclab package
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_target_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate and return the distance to the target.

    This function computes the Euclidean distance between the rover and the target.
    It then calculates a reward based on this distance, which is inversely proportional
    to the squared distance. The reward is also normalized by the maximum episode length.
    """

    # Accessing the target's position through the command manage,
    # we get the target position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and the reward
    distance = torch.norm(target_position, p=2, dim=-1)

    # Return the reward, normalized by the maximum episode length
    return (1.0 / (1.0 + (0.11 * distance * distance))) / env.max_episode_length
def reached_target(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """

    # Accessing the target's position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Get angle to target
    angle = env.command_manager.get_command(command_name)[:, 3]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)
    time_steps_to_goal = env.max_episode_length - env.episode_length_buf
    reward_scale = time_steps_to_goal / env.max_episode_length

    # Return the reward, scaled depending on the remaining time steps
    return torch.where((distance < threshold) & (torch.abs(angle) < 0.1), 2.0 * reward_scale, 0.0)
def oscillation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Calculate the oscillation penalty.

    This function penalizes the rover for oscillatory movements by comparing the difference
    in consecutive actions. If the difference exceeds a threshold, a squared penalty is applied.
    """
    # Accessing the rover's actions
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    # Calculating differences between consecutive actions
    linear_diff = action[:, 1] - prev_action[:, 1]
    angular_diff = action[:, 0] - prev_action[:, 0]

    # TODO combine these 5 lines into two lines
    angular_penalty = torch.where(
        angular_diff*3 > 0.05, torch.square(angular_diff*3), 0.0)
    linear_penalty = torch.where(
        linear_diff*3 > 0.05, torch.square(linear_diff*3), 0.0)

    angular_penalty = torch.pow(angular_penalty, 2)
    linear_penalty = torch.pow(linear_penalty, 2)

    return (angular_penalty + linear_penalty) / env.max_episode_length
def angle_to_target_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate the penalty for the angle between the rover and the target.

    This function computes the angle between the rover's heading direction and the direction
    towards the target. A penalty is applied if this angle exceeds a certain threshold.
    """

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    # Return the absolute value of the angle, normalized by the maximum episode length.
    return torch.where(torch.abs(angle) > 2.0, torch.abs(angle) / env.max_episode_length, 0.0)
def heading_soft_contraint(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate a penalty for driving backwards.

    This function applies a penalty when the rover's action indicates reverse movement.
    The penalty is normalized by the maximum episode length.
    """
    # return torch.where(env.action_manager.action[:, 0] < 0.0, (1.0 / env.max_episode_length), 0.0)
    reverse_speed = torch.clamp(-env.action_manager.action[:, 0], min=0.0)  # Only penalize negatives
    penalty = (reverse_speed * 0.4) / env.max_episode_length  # 0.2 is a tunable scale factor
    return penalty
def collision_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Calculate a penalty for collisions detected by the sensor.

    This function checks for forces registered by the rover's contact sensor.
    If the total force exceeds a certain threshold, it indicates a collision,
    and a penalty is applied.
    """
    # Accessing the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)
    # Calculating the force and applying a penalty if collision forces are detected
    normalized_forces = torch.norm(force_matrix, dim=1)
    forces_active = torch.sum(normalized_forces, dim=-1) > 1
    return torch.where(forces_active, 1.0, 0.0)
def far_from_target_reward(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Gives a penalty if the rover is too far from the target.
    """

    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    distance = torch.norm(target_position, p=2, dim=-1)

    return torch.where(distance > threshold, 1.0, 0.0)
def angle_to_goal_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate the angle to the goal.

    This function computes the angle between the rover's heading direction and the direction
    towards the goal. A reward is given based on the cosine of this angle.
    """
    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(target_vector_b, p=2, dim=-1)
    angle_b = env.command_manager.get_command(command_name)[:, 3]

    angle_reward = (1 / (1 + distance)) * 1 / (1 + torch.abs(angle_b))

    # Return the cosine of the angle, normalized by the maximum episode length.
    return angle_reward / env.max_episode_length
def forward_progress_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    action = env.action_manager.action[:, 0]  # linear velocity
    to_target = env.command_manager.get_command(command_name)[:, :2]

    angle = torch.atan2(to_target[:, 1], to_target[:, 0])
    alignment = torch.cos(angle)  # [-1, 1]

    forward_motion = torch.clamp(action, min=0.0)

    # Early-exploration shaping: strong incentive early, decays over episode
    early_bias = (1.0 - env.episode_length_buf / env.max_episode_length).clamp(min=0.1)

    # Positive: aligned forward motion
    positive_progress = forward_motion * torch.clamp(alignment, min=0.0)

    # Negative: forward motion in wrong direction (backward relative to goal)
    backward_penalty = forward_motion * torch.clamp(-alignment, min=0.0) * early_bias

    return (positive_progress - backward_penalty) / env.max_episode_length


@configclass
class RewardsCfg:
    distance_to_target = RewTerm(
        func=distance_to_target_reward,
        weight=6.0,
        params={"command_name": "target_pose"},
    )
    reached_target = RewTerm(
        func=reached_target,
        weight=10.0,
        params={"command_name": "target_pose", "threshold": 0.18},
    )
    oscillation = RewTerm(
        func=oscillation_penalty,
        weight=-0.05,
        params={},
    )
    angle_to_target = RewTerm(
        func=angle_to_target_penalty,
        weight=-0.5,
        params={"command_name": "target_pose"},
    )
    heading_soft_contraint = RewTerm(
        func=heading_soft_contraint,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg(name="robot")},
    )
    collision = RewTerm(
        func=collision_penalty,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_sensor"), "threshold": 1.0},
    )
    far_from_target = RewTerm(
        func=far_from_target_reward,
        weight=-1.0,
        params={"command_name": "target_pose", "threshold": 11.0},
    )
    relative_goal_orientation = RewTerm(
        func=angle_to_goal_reward,
        weight=2.0,
        params={"command_name": "target_pose"},
    )
    # progress_to_goal = RewTerm(
    #     func=forward_progress_reward,
    #     weight=4.0,
    #     params={"command_name": "target_pose"}
    # )

    @classmethod
    def from_yaml_dict(cls, yaml_cfg: dict) -> RewardsCfg:
        full_instance = cls()
        selected_terms = {}

        for key, val in vars(full_instance).items():
            if not isinstance(val, RewTerm):
                continue

            term_cfg = yaml_cfg.get(key, {})
            if not term_cfg.get("enabled", False):
                continue

            term = val.replace()
            if "weight" in term_cfg:
                term = term.replace(weight=term_cfg["weight"])
            if "threshold" in term_cfg and "threshold" in term.params:
                term.params["threshold"] = term_cfg["threshold"]

            selected_terms[key] = term

        rewards_cfg = cls()
        for k, v in selected_terms.items():
            setattr(rewards_cfg, k, v)
        return rewards_cfg