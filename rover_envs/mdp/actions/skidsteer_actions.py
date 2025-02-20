from __future__ import annotations

from typing import TYPE_CHECKING
import carb
import torch
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

    def __init__(self, cfg: actions_cfg.SkidSteerActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._drive_joint_ids, self, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        drive_order = cfg.drive_order
        sorted_drive_joint_names = sorted(self._left_wheel_joint_names, key=lambda x: drive_order.index(x[:2])) # Sorted drive joint names
        original_drive_id_positions = {name: i for i, name in enumerate(self._drive_joint_names)} # Origin positions for drive joints
        self._sorted_drive_ids = {self._drive_joint_ids[original_drive_id_positions[name]] for name in sorted_drive_joint_names}

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
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):
        """Apply computed wheel velocities to the robot."""
        left_wheel_vels, right_wheel_vels = self.skid_steer_simple(
            self._processed_actions[:, 0], self._processed_actions[:, 1], self.cfg, self.device
        )

        # Publish wheel velocities
        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._drive_joint_ids)

class SkidSteeringSimpleNonVec():
    def __init__(self,
        cfg:actions_cfg.SkidSteeringSimpleCfg,
        robot: Articulation,
        num_envs: int,
        device: torch.device):

        """ Initialize the SkidSteeringActionNonVec

        Args:
            cfg (actions_cfg.AckermannActionCfg): configuration for the ackermann action
            robot (Articulation): robot asset
            num_envs (int): number of environments
            device (torch.device): device to run the operation on
            """
        # Initialize parameters
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self._asset = robot

        # Find the joint ids and names for the drive and steering joints
        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        # Remap joints to the order specified in the config.
        drive_order = cfg.drive_order
        sorted_drive_joint_names = sorted(self._drive_joint_names, key=lambda x: drive_order.index(x[:2]))
        original_drive_id_positions = {name: i for i, name in enumerate(self._drive_joint_names)}
        self._sorted_drive_ids = [self._drive_joint_ids[original_drive_id_positions[name]]
                                  for name in sorted_drive_joint_names]
        
        carb.log_info(f" {self._drive_joint_ids} [{self._drive_joint_names}]")

        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)

        # Save the scale and offset for the actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2  # Assuming a 2D action vector (linear velocity, angular velocity)

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
        # Store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):
        # Apply the actions to the rover
        self._joint_vel = self.skid_steer_simple(
            self._processed_actions[:, 0], self._processed_actions[:, 1], self.cfg, self.device)

        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._sorted_drive_ids)


    def add_slippage(self, vx, omega, track_width, wheel_radius, cfg):
        """Accounting for slippage if it exists"""

        slip_factor = cfg.slip_factor
        effective_omega: float
        if slip_factor is not MISSING:
            effective_omega = omega * (1 - slip_factor)
        else:
            effective_omega = omega
        w_left = (vx - track_width * effective_omega) / wheel_radius
        w_right = (vx + track_width * effective_omega) / wheel_radius

        return w_left, w_right


    def skid_steer_simple(self, vx, omega, cfg, device):
        """Compute skid-steering wheel velocities."""

        track_width = cfg.track_width  # Track width (m)
        wheel_r = cfg.wheel_radius  # Wheel radius (m)

        w_left, w_right = self.add_slippage(vx, omega, track_width, wheel_r, cfg)

        return torch.stack([w_left, w_right, w_right, w_left], dim=1)  # Order: FL, FR, RR, RL