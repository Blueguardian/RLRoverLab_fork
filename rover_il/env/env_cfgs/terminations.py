from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# Importing necessary modules from the isaaclab package
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.envs.mdp import time_out

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def is_success(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """

    # Accessing the target's position
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    angle = env.command_manager.get_command(command_name)[:, 3]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)

    return torch.where((distance < threshold) & (torch.abs(angle) < 0.1), True, False)
    # return torch.where(distance < threshold, True, False)
def far_from_target(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """

    # Accessing the target's position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)

    return torch.where(distance > threshold, True, False)
def collision_with_obstacles(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Checks for collision with obstacles.
    """
    # Accessing the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Reshape as follows (num_envs, num_bodies, 3)
    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)

    # Calculating the force and returning true if it is above the threshold
    normalized_forces = torch.norm(force_matrix, dim=1)
    forces_active = torch.sum(normalized_forces, dim=-1) > 1

    return torch.where(forces_active, True, False)

@configclass
class TerminationsCfg:
    """Termination conditions for the task."""

    time_limit = DoneTerm(func=time_out, time_out=True)
    is_success = DoneTerm(
        func=is_success,
        params={"command_name": "target_pose", "threshold": 0.18},
    )
    far_from_target = DoneTerm(
        func=far_from_target,
        params={"command_name": "target_pose", "threshold": 11.0},
    )
    collision = DoneTerm(
        func=collision_with_obstacles,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_sensor"), "threshold": 1.0},
    )

    @classmethod
    def from_yaml_dict(cls, yaml_cfg: dict) -> TerminationsCfg:
        full_instance = cls()
        selected_terms = {}

        for key, val in vars(full_instance).items():
            if not isinstance(val, DoneTerm):
                continue

            term_cfg = yaml_cfg.get(key, {})
            if not term_cfg.get("enabled", False):
                continue

            term = val.replace()

            # Update threshold in params if provided
            if "threshold" in term_cfg and "threshold" in term.params:
                term.params["threshold"] = term_cfg["threshold"]

            selected_terms[key] = term

        # Construct and return
        termination_cfg = cls()
        for k, v in selected_terms.items():
            setattr(termination_cfg, k, v)
        return termination_cfg
