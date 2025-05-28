import argparse
import math
import os
import random
import sys
from datetime import datetime
import numpy as np
import wandb

import carb
import gymnasium as gym
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Isaac Lab: Omniverse Robotics Environments!")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="AAURoverEnvSimple-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--dagger", action="store_true",
                    help="Run the imitation‑learning DAgger pipeline "
                         "instead of the default SKRL RL training.")

parser.add_argument("--expert", type=str, default=None,
                    help="Absolute or relative path to the expert checkpoint "
                         "(required when you pass --dagger). "
                         "• .zip  → SB3 checkpoint is loaded directly\n"
                         "• .pt   → raw SKRL checkpoint is auto‑converted "
                         "to SB3 on‑the‑fly.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.dagger:
    if args_cli.expert is None:
        parser.error("--expert <checkpoint> must be provided when using --dagger")
    if args_cli.num_envs > 1:
        parser.error("--num_envs cannot be larger than one for this DAgger implementation")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

simulation_app = app_launcher.app

carb_settings = carb.settings.get_settings()
carb_settings.set_bool(
    "rtx/raytracing/cached/enabled",
    False,
)
carb_settings.set_int(
    "rtx/descriptorSets",
    8192,
)
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402


def log_setup(experiment_cfg, env_cfg, agent):
    """
    Setup the logging for the experiment.

    Note:
        Copied from the Isaac Lab framework.
    """
    # specify directory for logging experiments
    log_root_path = None
    if args_cli.dagger:
        log_root_path = os.path.join("logs", "dagger", experiment_cfg["agent"]["experiment"]["directory"])
    else:
        log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir = f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'

    log_dir += f"_{agent}"

    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)
    return log_dir


def video_record(
    env: ManagerBasedRLEnv, log_dir: str, video: bool, video_length: int, video_interval: int
) -> ManagerBasedRLEnv:
    """
    Function to check and setup video recording.

    Note:
        Copied from the Isaac Lab framework.

    Args:
        env (ManagerBasedRLEnv): The environment.
        log_dir (str): The log directory.
        video (bool): Whether or not to record videos.
        video_length (int): The length of the video (in steps).
        video_interval (int): The interval between video recordings (in steps).

    Returns:
        ManagerBasedRLEnv: The environment.
    """

    if video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        return gym.wrappers.RecordVideo(env, **video_kwargs)

    return env


from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402, F401

# Import agents
from rover_envs.envs.navigation.learning.skrl import get_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
if args_cli.dagger:
    import rover_il.agent.robots #noqa: E402, F401
else:
    import rover_envs.assets.robots  # noqa: E402, F401

def train():
    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs)
    # key = agent name, value = path to config file
    experiment_cfg_file = gym.spec(args_cli.task).kwargs.get("skrl_cfgs")[args_cli.agent.upper()]
    experiment_cfg = parse_skrl_cfg(experiment_cfg_file)

    log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

    # Create the environment
    render_mode = "rgb_array" if args_cli.video else None

    if args_cli.dagger:
        print("[INFO] Using dagger")

        from rover_il.DAgger.dagger_train import build_dagger
        from rover_il.DAgger.build_envs import make_shared_vecenv
        from rover_il.DAgger.expert_policy import SkrlExpertPolicy

        env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu",
                                num_envs=args_cli.num_envs)

        # build the shared vec envs
        vec_student, vec_expert = make_shared_vecenv(
            args_cli.task,
            cfg=env_cfg,
            video=args_cli.video
        )

        expert_policy = SkrlExpertPolicy(
            checkpoint=args_cli.expert,
            obs_space=vec_expert.observation_space,
            act_space=vec_expert.action_space,
            key_slices=vec_expert.key_slices,
            key_shapes=vec_expert.key_shapes,
        )

        dagger = build_dagger(
            vec_student=vec_student,
            vec_expert=vec_expert,
            expert_policy=expert_policy,
            logdir=log_dir,
        )

        dagger.train(total_timesteps=10_000, rollout_round_min_episodes=2, rollout_round_min_timesteps=500, bc_train_kwargs={"n_epochs": 2, "progress_bar": True})
        dagger.policy.save(os.path.join(log_dir, "policy"))# Clean up simulation and exit
        vec_student.close()
        vec_expert.close()
        simulation_app.close()
        return


    env = gym.make(args_cli.task, cfg=env_cfg, viewport=args_cli.video, render_mode=render_mode)
    # Check if video recording is enabled
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    # Wrap the environment
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])



    # Get the observation and action spaces
    num_obs = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.unwrapped.action_manager.action_term_dim[0]
    observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

    # policy_names = env.unwrapped.observation_manager._group_obs_term_names["policy"]
    # policy_dims = env.unwrapped.observation_manager._group_obs_term_dim["policy"]
    #
    # term_shape_map = dict(zip(policy_names, policy_dims))

    # observation_space = gym.spaces.Dict({
    #     "camera_rgb": gym.spaces.Box(low=0, high=255, shape=term_shape_map["camera_rgb"], dtype=np.uint8),
    #     "camera_depth": gym.spaces.Box(low=0, high=10, shape=term_shape_map["camera_depth"], dtype=np.float32),
    #     "heading": gym.spaces.Box(low=-math.inf, high=math.inf, shape=term_shape_map["heading"]),
    #     "distance": gym.spaces.Box(low=-math.inf, high=math.inf, shape=term_shape_map["distance"]),
    #     "relative_goal_orientation": gym.spaces.Box(low=-math.inf, high=math.inf, shape=term_shape_map["relative_goal_orientation"]),
    #     "actions_taken": gym.spaces.Box(low=-1.0, high=1.0, shape=term_shape_map["actions"]),
    # })

    trainer_cfg = experiment_cfg["trainer"]

    agent = get_agent(args_cli.agent, env, observation_space, action_space, experiment_cfg, conv=True)
    trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.train()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train()
