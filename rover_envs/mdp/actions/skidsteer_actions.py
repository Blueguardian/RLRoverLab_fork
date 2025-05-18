from __future__ import annotations

from typing import TYPE_CHECKING
import carb
import torch
import re
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from dataclasses import MISSING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv  # noqa: F811
    from . import actions_cfg


class SkidSteerAction(ActionTerm):
    """
    Action term for controlling a skid-steered mobile robot
    Should be considered similar to
    """

    cfg: actions_cfg.SkidSteeringSimpleCfg
    _asset: Articulation

    _wheel_radius: float
    _track_width: float

    _drive_joint_names: list[str]

    _scale: torch.Tensor
    _offset: torch.Tensor

    

    def __init__(self, cfg: actions_cfg.SkidSteeringSimpleCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._dummy_joint_names = self._asset.joint_names
        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)
        self._bogie_joint_ids, self._bogie_joint_names = self._asset.find_joints(self.cfg.bogie_joint_names)

        drive_order = cfg.drive_order
        sorted_drive_joint_names = sorted(self._drive_joint_names, key=lambda x: drive_order.index(x[:2]))

        original_drive_id_positions = {name: i for i, name in enumerate(self._drive_joint_names)} # Origin positions for drive joints
        self._sorted_drive_ids = [self._drive_joint_ids[original_drive_id_positions[name]]
                                  for name in sorted_drive_joint_names]

        if self.cfg.drive_id_override is not None:
            self._sorted_drive_ids = self.cfg.drive_id_override

        # Define tensors for actions and joint velocities
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)

        # Scale and offset for action processing
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)


    @property
    def action_dim(self) -> int:
        return 2  # Linear velocity (m/s), Angular velocity (rad/s)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        """Process raw actions into velocities for the wheels."""

        # actions[:] =
        self._raw_actions[:] = actions

        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):
        """Apply computed wheel velocities to the robot."""
        self._joint_vel = skid_steer_simple(
            self._processed_actions[:, 0], self._processed_actions[:, 1], self.cfg, self.device
        )
        # Publish wheel velocities
        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._sorted_drive_ids)



def skid_steer_simple(vx, omega, cfg, device):
    """Compute skid-steering wheel velocities."""
    #
    # vx[:][:] = 0
    # omega[:][:] = -2

    # Instance configuration variables
    track_width = cfg.track_width  # Track width (m)
    wheel_r = cfg.wheel_radius  # Wheel radius (m)

    lin_vel = torch.abs(vx)

    vel_left = vx - (omega * track_width / 2) / wheel_r
    vel_right = vx + (omega * track_width / 2) / wheel_r

    # wheel_vel = torch.stack([vel_left, vel_right, vel_left, vel_right], dim=1)  # Order: FL, RL, FL, RR -> Leo rover
    wheel_vel = torch.stack([vel_left, vel_left, vel_right, vel_right], dim=1) # Order FL, FR, RL, RR -> Summit

    return wheel_vel