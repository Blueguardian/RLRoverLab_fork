import torch
import numpy as np
import torch.nn as nn
from gymnasium.spaces import Dict, Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rover_envs.envs.navigation.learning.skrl.models import ConvHeightmapEncoder, ResnetEncoder

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(0.2, inplace=True)
    raise KeyError(name)

class SB3_GaussianNeuralNetworkConv(BaseFeaturesExtractor):
    """SB3 feature extractor replicating GaussianNeuralNetworkConv from SKRL."""

    def __init__(self,
                 observation_space: Dict,
                 device: str = "cuda:0",
                 mlp_layers: list=[256, 160],
                 mlp_activation: str = "leaky_relu",
                 encoder_input_size: int = None,
                 encoder_layers: list = [80, 60],
                 encoder_activation: str = "leaky_relu",
                 features_dim: int = 128,
                 actor: str = "teacher",
                 **kwargs):
        super().__init__(observation_space, features_dim)

        self._device = device
        self._mlp_layers = mlp_layers.append(features_dim)
        self._mlp_activation = mlp_activation
        self._encoder_input_size = encoder_input_size
        self._encoder_layers = encoder_layers
        self._encoder_activation = encoder_activation
        self._features_dim = features_dim

        # Calling original Heightmap encoder
        self._in_channels = encoder_input_size
        if self._encoder_input_size is not None:
            self.encoder = ConvHeightmapEncoder(
                in_channels=self._in_channels,
                encoder_features=encoder_input_size,
                encoder_activation=get_activation(encoder_activation),
            )
            self._in_channels += self.encoder.out_features

        # Collect all proprioceptive observation keys
        self.proprio_keys = [key for key in observation_space.keys()
                             if actor in key
                             and "height_scan" not in key
                             ]
        self._in_channels += sum(observation_space[k].shape[0] for k in self.proprio_keys)

        # Instantiate the MLP object
        self.mlp = nn.ModuleList()

        # Create the layer defined by mlp_layers
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(self._in_channels, feature))
            self.mlp.append(get_activation(get_activation(mlp_activation)))
            self._in_channels += feature

        # Add the final output layer
        self.mlp.append(nn.Linear(self._in_channels, features_dim))
        self.mlp.append(get_activation(mlp_activation))

    def forward(self, obs):
        # Encode inputs
        proprio = torch.cat([obs[k] for k in self.proprio_keys], dim=1)
        height = obs["height_scan_teacher"]
        height_feat = self.encoder(height)

        x = torch.cat([proprio, height_feat], dim=1)
        for layer in self.mlp:
            x = layer(x)
        return x


# rover_il/learning/models/models.py
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SB3_GaussianNeuralNetworkConvResNet(BaseFeaturesExtractor):
    """
    features_extractor for SB3/PPO when your obs is a single flat Box of length FLAT.
    """
    def __init__(
        self,
        observation_space,            # gym.spaces.Box(shape=(flat_dim,))
        *,
        key_slices: dict,             # from FlattenPolicyObs
        key_shapes: dict,             # from FlattenPolicyObs
        encoder_layers=(80, 60),
        encoder_activation=("relu","leaky_relu"),
        mlp_layers=(256,160),
        mlp_activation="leaky_relu",
        features_dim=128,
    ):
        super().__init__(observation_space, features_dim)
        self.slices     = key_slices
        self.shapes     = key_shapes

        # build your two vision encoders:
        self.encoder_rgb = ResnetEncoder(
            in_channels=3,
            encoder_features=list(encoder_layers),
            encoder_activation=encoder_activation[0],
        )
        self.encoder_depth = ConvHeightmapEncoder(
            in_channels=10000,
            encoder_features=list(encoder_layers),
            encoder_activation=encoder_activation[1],
        )

        # figure out which keys are proprio (non-camera) for student:
        self.proprio_keys = [
            k for k in key_slices
            if k.endswith("_student") and "camera" not in k
        ]

        # mlp input dim = rgb_feat + depth_feat + sum(proprio dim)
        in_dim = self.encoder_rgb.out_features + self.encoder_depth.out_features
        in_dim += sum(int(np.prod(self.shapes[k])) for k in self.proprio_keys)

        # build MLP
        mlp = []
        for h in mlp_layers:
            mlp += [nn.Linear(in_dim, h), get_activation(mlp_activation)]
            in_dim = h
        mlp += [nn.Linear(in_dim, features_dim)]
        self.mlp = nn.Sequential(*mlp)

    def _slice(self, flat: torch.Tensor, key: str):
        sl  = self.slices[key]
        shp = (flat.size(0),) + self.shapes[key]
        return flat[:, sl].view(shp)

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        # flat_obs: (batch=num_envs, flat_dim)
        # --- vision ---
        rgb   = self._slice(flat_obs,   "camera_rgb_student")   # (B,3,H,W)
        depth = self._slice(flat_obs, "camera_depth_student")   # (B,1,H,W)
        rgb_f = self.encoder_rgb(rgb)
        dpt_f = self.encoder_depth(depth)

        # --- proprio ---
        proprio = [
            self._slice(flat_obs, k).view(flat_obs.size(0), -1)
            for k in self.proprio_keys
        ]
        pr_f = torch.cat(proprio, dim=1)

        x = torch.cat([rgb_f, dpt_f, pr_f], dim=1)              # (B, total_feat)
        x = self.mlp(x)  # (B, features_dim)

        return x
