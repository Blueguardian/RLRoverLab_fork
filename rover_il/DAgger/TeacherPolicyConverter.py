
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.policies import MultiInputPolicy
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
import torch.nn as nn
import numpy as np
import torch
from pathlib import Path

from rover_envs.envs.navigation.learning.skrl.models import GaussianNeuralNetworkConv

class TanhMultiInputPolicy(MultiInputPolicy):
    def make_actor(self, features_dim, net_arch, activation_fn, squash_output, log_std_init):
        actor_net = nn.Sequential(
            nn.Linear(features_dim, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], self.action_space.shape[0]),
            nn.Tanh()
        )
        return actor_net



class SkrlExpertPolicyWrapper(BasePolicy):
    def __init__(self, skrl_model, observation_keys, action_space, observation_space, device="cuda:0"):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=None,
        )
        self.model = skrl_model
        self._device = device
        self.observation_keys = observation_keys

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Not used.")

    def _predict(self, obs, deterministic=True):
        input_tensor = torch.cat([obs[k] for k in self.observation_keys], dim=1).to(self._device)
        with torch.no_grad():
            action = self.model(input_tensor)
        return action, None


class DAggerPolicyExpert:
    """Wraps and loads a SKRL PPO teacher for DAgger training."""

    def __init__(
        self,
        env,
        expert_path: str,
        framework: str = "skrl",
        min_timesteps: int = 10_000,
        min_episodes: int = 10,
    ):
        print(f"[DEBUG] env input: {env.observation_space}")

        # Extract teacher obs before wrapping

        self._obs_space = env.observation_space["policy"]
        self._obs_keys = [k for k in self._obs_space.spaces if "teacher" in k]
        env.observation_space = self._obs_space
        # Clip action space to match SB3/DAgger expectation
        env.action_space = spaces.Box(
            low=np.full_like(env.action_space.low, -1.0),
            high=np.full_like(env.action_space.high, 1.0),
            dtype=env.action_space.dtype,
        )

        # Wrap in DummyVecEnv after extracting obs keys
        self.env = DummyVecEnv([lambda: env])
        self._input_framework = framework
        self._expert_path = expert_path
        self._min_timesteps = min_timesteps
        self._min_episodes = min_episodes

        # Load + wrap the SKRL model
        self.teacher = self.convert_expert_to_sb3(self.env, self._expert_path)
        if self.teacher is None:
            raise RuntimeError("Expert model could not be loaded.")
        self._policy = self.teacher

    def convert_expert_to_sb3(self, env, expert_ckpt_path: str, device="cuda:0"):
        try:
            expert_path = Path(expert_ckpt_path)
            if not expert_path.is_absolute():
                repo_root = Path(__file__).resolve().parents[2]
                expert_path = (repo_root / expert_ckpt_path).resolve()

            if not expert_path.exists():
                raise FileNotFoundError(f"Missing: {expert_path}")
            model = GaussianNeuralNetworkConv(
                observation_space=self._obs_space,
                action_space=env.action_space,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                actor="teacher"
            )

            state_dict = torch.load(expert_path, map_location=device)
            print("[DEBUG] Top-level keys:", list(state_dict.keys()))

            if isinstance(state_dict, dict) and "policy" in state_dict and isinstance(state_dict["policy"], dict):
                state_dict = state_dict["policy"]
                print("[DEBUG] Extracted SKRL 'policy' dict:", list(state_dict.keys()))

            model.load_state_dict(state_dict)
            model.eval()
            print(f"[SKRL Expert loaded] from {expert_path}")

            return SkrlExpertPolicyWrapper(
                skrl_model=model,
                observation_keys=self._obs_keys,
                action_space=env.action_space,
                observation_space=self._obs_space,
                device=device,
            )

        except Exception as e:
            print(f"[Expert conversion failed]: {e}")
            return None

    @property
    def input_framework(self):
        return self._input_framework

    @property
    def env(self):
        return self._env

    @property
    def policy(self):
        return self._policy

    @property
    def expert_path(self):
        return self._expert_path

    @property
    def min_timesteps(self):
        return self._min_timesteps

    @property
    def min_episodes(self):
        return self._min_episodes

    @env.setter
    def env(self, value):
        self._env = value

    @min_timesteps.setter
    def min_timesteps(self, value):
        self._min_timesteps = value

    @min_episodes.setter
    def min_episodes(self, value):
        self._min_episodes = value



