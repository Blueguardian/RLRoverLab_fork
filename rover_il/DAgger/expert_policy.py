# rover_il/DAgger/expert_policy.py
# --------------------------------------------------------------------- #
import numpy as np, torch
from pathlib import Path
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from rover_envs.envs.navigation.learning.skrl.models import GaussianNeuralNetworkConv
# from rover_il.learning.models.skrl_models import (
#     GaussianNeuralNetworkConv,
# )

TEACHER_KEYS = [
    "distance_teacher",
    "heading_teacher",
    "relative_goal_orientation_teacher",
    "actions_teacher",
    "height_scan_teacher",
]

# --------------------------------------------------------------------- #
class SkrlExpertPolicy(BasePolicy):
    """
    Wrap the original SKRL Gaussian network so SB-3 / imitation can call
    `.predict()`.  We pass in `key_slices` (dict: name→slice) so the wrapper
    can extract exactly the teacher features from the *flat* observation.
    """

    def __init__(
        self,
        *,
        checkpoint   : str,
        obs_space    : spaces.Box,      # flat Box
        act_space    : spaces.Box,
        key_slices   : dict[str, slice], # from FlattenPolicyObs
        key_shapes   : dict[str, int],
        device       : str = "cuda:0",
    ):
        super().__init__(
            observation_space = obs_space,
            action_space      = act_space,
            features_extractor= None,
        )
        self.key_slices = key_slices     # keep mapping
        self.key_shapes = key_shapes

        self.net = GaussianNeuralNetworkConv(
            observation_space = spaces.Box(-np.inf, np.inf, shape=(10206,), dtype=np.float32),
            action_space      = spaces.Box(low=-1, high=1, shape=(2,),    dtype=np.float32),
            device            = self.device,
            mlp_input_size    = 5,
            encoder_input_size= 10201,
            encoder_layers    = [8, 16, 32, 64],
            encoder_activation= "leaky_relu",
            mlp_layers=[256, 160, 128],
            mlp_activation="leaky_relu",
        ).to(self.device)

        ckpt = Path(checkpoint)
        if not ckpt.is_absolute():
            ckpt = (Path(__file__).resolve().parents[2] / checkpoint).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)

        sd = torch.load(ckpt, map_location=self.device, weights_only=True)
        missing, unexpected = self.net.load_state_dict(sd["policy"] if "policy" in sd else sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: {missing=} {unexpected=}")
        self.net.eval()

        # ----------------- cache teacher-only slice ------------------------ #
        # (distance, heading, rel_goal, actions, height_scan) → 133 dims
        teach_keys = [
            "distance_teacher",
            "heading_teacher",
            "relative_goal_orientation_teacher",
            "actions_teacher",
            "height_scan_teacher",
        ]
        idx = [self.key_slices[k] for k in teach_keys]  # teacher slices
        # flatten the slices into one continuous list of indices
        teacher_indices = np.concatenate(
            [np.arange(s.start, s.stop, dtype=np.int64) for s in idx]
        )
        self.teacher_indices = torch.as_tensor(teacher_indices, dtype=torch.long)
        self.net.eval()

        print("first 10 indices in policy     :", self.teacher_indices[:10])

    # ------------------------------------------------------------------ #
    #  SB-3 / imitation calls this through policy.predict()               #
    # ------------------------------------------------------------------ #
    def _predict(self, obs: np.ndarray, deterministic: bool = True):
        if obs.ndim == 1:  # allow single obs
            obs = obs[None, :]

        # 1) grab all teacher columns (len = 10 206)
        flat = torch.as_tensor(obs, device=self.device)  # (B, 50 211)
        flat = flat.index_select(-1, self.teacher_indices.to(flat.device))

        # 2) build the **clean** vector: keep 0‥4, skip 4 → 5, keep the rest
        clean = torch.cat([flat[..., :5], flat[..., 5:]], dim=-1)  # (B, 10 206)
        assert clean.shape[-1] == 10_206  # safety

        # 3) forward through the ORIGINAL network
        with torch.no_grad():
            mean, log_std, _ = self.net({"states": clean})

            if deterministic:
                act = mean
            else:  # match SKRL training behaviour
                std = log_std.exp()
                eps = torch.randn_like(mean)
                act = torch.tanh(mean + eps * std)

        return act
