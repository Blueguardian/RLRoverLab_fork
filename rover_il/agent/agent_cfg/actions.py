from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg as ActionTerm
from dataclasses import MISSING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@configclass
class ActionsCfg:
    """Action"""

    # We define the action space for the rover
    actions: ActionTerm = MISSING