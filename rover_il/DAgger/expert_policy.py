# rover_il/DAgger/expert_policy.py
# --------------------------------------------------------------------- #
import numpy as np, torch
from pathlib import Path
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from rover_il.learning.models.skrl_models import (
    GaussianNeuralNetworkConv,
)

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

        # ------------------------------------------------------------------ #
        #  Build the original SKRL network (133-D proprio + 10201 heightmap) #
        # ------------------------------------------------------------------ #
        self.net = GaussianNeuralNetworkConv(
            observation_space = spaces.Box(-np.inf, np.inf, shape=(133,), dtype=np.float32),
            action_space      = spaces.Box(-1, 1, shape=(2,),    dtype=np.float32),
            device            = self.device,
            mlp_input_size    = 5,
            encoder_input_size= 10201,
            encoder_layers    = [8, 16, 32, 64],
            encoder_activation= "leaky_relu",
        ).to(self.device)

        ckpt = Path(checkpoint)
        if not ckpt.is_absolute():
            ckpt = (Path(__file__).resolve().parents[2] / checkpoint).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)

        sd = torch.load(ckpt, map_location=self.device)
        self.net.load_state_dict(sd["policy"] if "policy" in sd else sd, strict=False)
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
        idx = [self.key_slices[k] for k in teach_keys]
        # concatenate consecutive ranges into one slice for speed
        start = idx[0].start
        stop  = idx[-1].stop
        self.teacher_slice = slice(start, stop)

    # ------------------------------------------------------------------ #
    #  SB-3 / imitation calls this through policy.predict()               #
    # ------------------------------------------------------------------ #
    def _predict(self, obs: np.ndarray, deterministic: bool = True):
        """
        SB3 will call .predict(obs) where obs is a (batch,flat_dim) np.ndarray.
        We just run it through our SKRL net and return (batch,act_dim).
        """
        if obs.ndim == 1:
            obs = obs[None, :]
        batch = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions_mean, _, _ = self.net({"states": batch})
        return actions_mean
