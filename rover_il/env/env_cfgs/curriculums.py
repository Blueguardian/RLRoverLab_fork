from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def gradual_change_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    min_weight: float,
    max_weight: float,
    start_step: int,
    end_step: int,
):
    """
    Linearly scale the reward weight from min_weight to max_weight between start_step and end_step.

    Args:
        env: The environment instance.
        env_ids: (unused) Environment IDs.
        term_name: The name of the reward term.
        min_weight: Initial weight value.
        max_weight: Final weight value.
        start_step: Step at which scaling begins.
        end_step: Step at which scaling ends.
    """
    step = env.common_step_counter

    if step < start_step:
        new_weight = min_weight
    elif step > end_step:
        new_weight = max_weight
    else:
        ratio = (step - start_step) / (end_step - start_step)
        new_weight = min_weight + ratio * (max_weight - min_weight)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)


@configclass
class CurriculumCfg:
    """ Curriculum configuration for the task. """
    # collision = CurrTerm(
    #     func=gradual_change_reward_weight,
    #     params={
    #         "term_name": "collision",
    #         "min_weight": -2.0,
    #         "max_weight": -6.0,
    #         "start_step": 50000,
    #         "end_step": 300000}
    # )
    # far_from_target = CurrTerm(
    #     func=gradual_change_reward_weight,
    #     params={
    #         "term_name": "far_from_target",
    #         "min_weight": -0.5,
    #         "max_weight": -3.0,
    #         "start_step": 100_000,
    #         "end_step": 300_000
    #     }
    # )
    # distance_to_target = CurrTerm(
    #     func=gradual_change_reward_weight,
    #     params={
    #         "term_name": "distance_to_target",
    #         "min_weight": 12.0,
    #         "max_weight": 6.0,
    #         "start_step": 0,
    #         "end_step": 300_000
    #     }
    # )
    # angle_to_target = CurrTerm(
    #     func=gradual_change_reward_weight,
    #     params={
    #         "term_name": "angle_to_target",
    #         "min_weight": -0.05,
    #         "max_weight": -0.5,
    #         "start_step": 0,
    #         "end_step": 150_000
    #     }
    # )