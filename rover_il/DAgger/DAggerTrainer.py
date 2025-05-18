import os
from gymnasium import spaces
import torch.nn as nn
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import DAggerTrainer, SimpleDAggerTrainer
from stable_baselines3.common.policies import BasePolicy


class DataAggregationTrainer:
    """DAgger trainer for RGB-D student using SB3 PPO expert (already loaded)."""
    def __init__(
        self,
        env,
        expert: BasePolicy,  # passed from caller, already converted
        total_steps: int = 100_000,
        device: str = "cuda:0",
        log_dir: str = "./logs/dagger",
    ):
        self._env = env
        self.expert = expert  # ✅ ALREADY converted externally
        self.total_steps = total_steps
        self.device = device
        self.log_dir = log_dir
        self._rng = np.random.default_rng(0)
        # 1. Instantiate fresh SB3 PPO student (no pretrained weights)
        self.student = PPO(
            policy="MultiInputPolicy",
            env=self._env,
            device=self.device,
            verbose=1,
        )
        # 2. Wrap student in a BC trainer (used internally by DAgger)
        self.bc_trainer = BC(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
            policy=self.student.policy,
            rng=self._rng,
            demonstrations=None,  # updated automatically per round
            device=self.device,
        )
        # 3. DAgger trainer (uses internal collector, β = 1.0)
        self.dagger_trainer = SimpleDAggerTrainer(
            venv=self._env,
            expert_policy=self.expert,
            scratch_dir=self.log_dir,
            rng=self._rng,
            beta_schedule=lambda _: 1.0,
            bc_trainer=self.bc_trainer,
        )

    def train(self):
        print(f"[DAggerTrainer] Starting DAgger training for {self.total_steps} env steps")
        self.dagger_trainer.train(total_timesteps=self.total_steps)
        ckpt_path = os.path.join(self.log_dir, "student_dagger.pt")
        torch.save(self.student.policy.state_dict(), ckpt_path)
        print(f"[DAggerTrainer] Saved student policy to {ckpt_path}")