# rover_il/DAgger/wrappers.py
# --------------------------------------------------------------------- #
#  Tiny utility wrappers used by the DAgger pipeline                    #
# --------------------------------------------------------------------- #
import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Tuple, List


# --------------------------------------------------------------------- #
#  Helper: recursively turn torch.Tensor → contiguous np.ndarray        #
# --------------------------------------------------------------------- #
def _as_numpy(x):
    """Detach → CPU → contiguous → numpy.  Recurse over dicts."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().contiguous().numpy()
    if isinstance(x, dict):
        return {k: _as_numpy(v) for k, v in x.items()}
    return x  # already np, list, scalar, …

# --------------------------------------------------------------------- #
#  Action-space wrapper                                                 #
# --------------------------------------------------------------------- #
class FiniteActionBox(gym.Wrapper):
    """
    SB-3 requires every continuous Box to have finite bounds.
    We clamp ±∞ to ±1 (configurable) and clip outgoing actions.
    """
    def __init__(self, env: gym.Env, lo: float = -1.0, hi: float = 1.0):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), \
            "FiniteActionBox works only on Box action spaces"

        self.orig_space = env.action_space
        low  = np.full_like(self.orig_space.low,  lo, dtype=np.float32)
        high = np.full_like(self.orig_space.high, hi, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        action = _as_numpy(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return super().step(action)

# --------------------------------------------------------------------- #
#  Observation flattening                                               #
# --------------------------------------------------------------------- #
# Order matters: must match student-extractor reconstruction
TEACHER_KEYS: List[str] = [
    "distance_teacher",
    "heading_teacher",
    "relative_goal_orientation_teacher",
    "actions_teacher",
    "height_scan_teacher",
]

STUDENT_KEYS: List[str] = [
    "distance_student",
    "heading_student",
    "relative_goal_orientation_student",
    "actions_student",
    "camera_depth_student",
    "camera_rgb_student",
]

class FlattenPolicyObs(gym.ObservationWrapper):
    """
    Replace `obs["policy"]` (a nested Dict of Boxes) by a *single*
    flat `Box(N,)`.

    The mapping information (`key_slices`, `key_shapes`) is stored as
    attributes so external code (student feature-extractor) can undo
    the flattening.
    """

    def __init__(self, env: gym.Env, keys: List[str]):
        super().__init__(env)
        self.counter = 0 # TODO: Remove after image extraction
        # ----------------------------------------------------------------- #
        #  Resolve inner "policy" space                                     #
        # ----------------------------------------------------------------- #
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError(
                "Environment must provide Dict observation with a 'policy' key"
            )
        policy_space: gym.spaces.Dict = env.observation_space["policy"]  # type: ignore[index]

        # ----------------------------------------------------------------- #
        #  Build slice table                                                #
        # ----------------------------------------------------------------- #
        self.keys         = list(keys)
        self.key_shapes: Dict[str, Tuple[int, ...]] = {
            k: policy_space[k].shape[1:] if len(policy_space[k].shape) > 1 else ()
            for k in self.keys
        }

        start = 0
        self.key_slices: Dict[str, slice] = {}
        for k in self.keys:
            size = int(np.prod(self.key_shapes[k]))
            self.key_slices[k] = slice(start, start + size)
            start += size

        # Resulting flat space
        flat_low  = -np.inf * np.ones(start, dtype=np.float32)
        flat_high =  np.inf * np.ones(start, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=flat_low, high=flat_high, dtype=np.float32)

    def observation(self, obs):
        inner = obs["policy"]
        parts = [_as_numpy(inner[k]).reshape(-1) for k in self.keys]
        return np.concatenate(parts, axis=0).astype(np.float32)


# --------------------------------------------------------------------- #
#  Side-specific convenience wrappers                                   #
# --------------------------------------------------------------------- #
class TorchTensorToNumpy(gym.Wrapper):
    """
    After `FlattenPolicyObs` the observation is already 1-D **and**
    contains only numpy for pixels but tensors for other terms
    (Isaac Sim returns torch).  Convert *everything* to numpy so that
    `imitation` assertions (`np.ndarray or dict`) pass.
    """
    def reset(self, **kw):
        obs, info = super().reset(**kw)
        return _as_numpy(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        return _as_numpy(obs), rew, term, trunc, info


class StudentIdentity(gym.Wrapper):
    """
    No-op. We keep it for symmetry (in case you want to hook logging,
    normalisation, … only on the student side later).
    """
    pass

class NumpyToTorchAction(gym.Wrapper):
    """
    Convert the incoming action (np.ndarray) into a torch.Tensor on the right device,
    so that downstream Isaac-Lab envs (which do `action.to(self.device)`) will work.
    """
    def __init__(self, env, device="cuda:0", dtype=torch.float32):
        super().__init__(env)
        self.device = torch.device(device)
        self.dtype  = dtype

    def step(self, action: np.ndarray):
        # action: np.ndarray of shape matching env.action_space
        t = torch.as_tensor(action, dtype=self.dtype, device=torch.device("cuda:0"))
        # now pass the torch tensor down into the sim
        return super().step(t)