# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Utility functions for parsing skrl configuration files."""

import yaml

__all__ = ["parse_skrl_cfg"]


def parse_skrl_cfg(task_name) -> dict:
    """Parse configuration based on command line arguments.

    Args:
        task_name (str): The name of the environment.

    Returns:
        dict: A dictionary containing the parsed configuration.
    """

    # retrieve the default environment config file
    try:
        #config_file = SKRL_PPO_CONFIG_FILE[task_name]
        config_file = task_name
    except KeyError:
        raise ValueError(f"Task not found: {task_name}")

    # parse agent configuration
    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)  # noqa: R504
    return cfg  # noqa: R504


def convert_skrl_cfg(cfg):
    """Convert simple YAML types to skrl classes/components.

    Args:
        cfg (dict): configuration dictionary.

    Returns:
        dict: A dictionary containing the converted configuration.
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "state_preprocessor",
        "value_preprocessor",
        "input_shape",
        "output_shape",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, timestep, timesteps):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)

        return d

    # parse agent configuration and convert to classes
    return update_dict(cfg)
