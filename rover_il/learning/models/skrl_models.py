# import torch, torch.nn as nn, numpy as np
# from skrl.models.torch.base import Model as BaseModel
# from skrl.models.torch.gaussian import GaussianMixin
# from rover_envs.envs.navigation.learning.skrl.models import (
#     ConvHeightmapEncoder, get_activation
# )
#
# class GaussianNNFlat(GaussianMixin, BaseModel):
#     """
#     Expert that *re-assembles* pieces from a flat vector using key_slices/shapes.
#     """
#
#     def __init__(self, *, key_slices, key_shapes,
#                  action_space, device,
#                  mlp_layers=(256,160,128), mlp_activation="leaky_relu",
#                  encoder_layers=(80,60), encoder_activation="leaky_relu"):
#
#         # dummy observation_space just for BaseModel API
#         obs_len = max(sl.stop for sl in key_slices.values())
#         dummy_obs = None                                  # never queried
#         BaseModel.__init__(self, dummy_obs, action_space, device)
#         GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True,
#                                min_log_std=-20, max_log_std=2, reduction="sum")
#
#         self.slices  = key_slices
#         self.shapes  = key_shapes
#         self.device  = device
#
#         # ---- which parts feed the encoder vs. proprio --------------------
#         scalars_sl  = key_slices["distance_teacher"]      # first scalar slice
#         mlp_input   = scalars_sl.stop                     # assume teacher scalars first
#         enc_sl      = key_slices["height_scan_teacher"]   # height map slice
#
#         # height-map encoder
#         self.encoder = ConvHeightmapEncoder(
#             in_channels=key_shapes["height_scan_teacher"][0],
#             encoder_features=encoder_layers,
#             encoder_activation=encoder_activation,
#         )
#
#         in_dim = mlp_input + self.encoder.out_features
#         layers = []
#         for h in mlp_layers:
#             layers += [nn.Linear(in_dim, h), get_activation(mlp_activation)]
#             in_dim = h
#         layers += [nn.Linear(in_dim, action_space.shape[0]), nn.Tanh()]
#         self.mlp = nn.Sequential(*layers)
#
#         self.log_std_parameter = nn.Parameter(
#             torch.zeros(action_space.shape[0])
#         )
#
#     # ---------------------------------------------------------------------
#     def compute(self, flat, role="actor"):
#         B = flat.size(0)
#
#         scalars = flat[:, : self.slices["distance_teacher"].stop]          # (B,5)
#         hmap    = flat[:, self.slices["height_scan_teacher"]]              # (B,10201)
#
#         enc     = self.encoder(hmap)
#         x       = torch.cat([scalars, enc], dim=1)
#         mu      = self.mlp(x)
#         return mu, self.log_std_parameter, {}

# rover_il/learning/models/gaussian_conv_flat.py
# ------------------------------------------------
import torch
import torch.nn as nn
from skrl.models.torch.base     import Model as BaseModel
from skrl.models.torch.gaussian import GaussianMixin
from rover_envs.envs.navigation.learning.skrl.models import (
    ConvHeightmapEncoder,
    get_activation,
)

class GaussianNeuralNetworkConv(GaussianMixin, BaseModel):
    """
    A flat, single-vector Gaussian network that SKRL can call with
    inputs={"states": Tensor(B,10206)} and that returns exactly
    (actions_mean, log_std, {}) with shapes [(B,2), (2,), {}].
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size      = 5,        # the first 5 dims: dist, head, rel_orient, a0, a1
        mlp_layers          = (256,160,128),
        mlp_activation      = "leaky_relu",
        encoder_input_size  = 10201,    # the remaining dims: height_scan_teacher
        encoder_layers      = (8,16,32,64),
        encoder_activation  = "leaky_relu",
        **kwargs,
    ):
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions   = True,
            clip_log_std   = True,
            min_log_std    = -20.0,
            max_log_std    =   2.0,
            reduction      = "sum",
        )

        self.mlp_input_size     = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_dim = mlp_input_size
        # build the height‐scan encoder if requested
        if encoder_input_size:
            self.encoder = ConvHeightmapEncoder(
                in_channels=encoder_input_size,
                encoder_features=list(encoder_layers),
                encoder_activation=encoder_activation
            )
            in_dim += self.encoder.out_features

        # a simple MLP head
        layers = []
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), get_activation(mlp_activation)]
            in_dim = h
        # final 2‐D action + Tanh
        layers += [nn.Linear(in_dim, action_space.shape[0]), nn.Tanh()]
        self.mlp = nn.Sequential(*layers)

        # SKRL requires a learned log-std parameter
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space.shape[0]))

    def forward(self, inputs, role="actor"):
        """
        SKRL will call forward(inputs, role), where inputs is a dict
        with key "states" → Tensor(B, 10206).
        """
        # pull out the raw tensor (B,10206)
        if isinstance(inputs, dict) and "states" in inputs:
            x = inputs["states"]
        else:
            x = inputs  # just in case SKRL ever calls with raw Tensor

        # slice off the first 5 dims for the MLP
        proprio = x[:, : self.mlp_input_size]  # (B,5)

        # slice off the next encoder_input_size dims for the encoder
        if self.encoder_input_size:
            start = self.mlp_input_size
            end   = start + self.encoder_input_size
            height = x[:, start:end]           # (B,10201)
            enc_out = self.encoder(height)     # (B, encoder_dim)
            feat_in = torch.cat([proprio, enc_out], dim=1)
        else:
            feat_in = proprio

        # pass through the MLP
        actions_mean = self.mlp(feat_in)        # (B,2)

        # return exactly the triple that SKRL expects
        return actions_mean, self.log_std_parameter, {}