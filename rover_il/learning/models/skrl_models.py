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

        self.mlp = nn.ModuleList()
        for layer in mlp_layers:
            self.mlp.append(nn.Linear(in_dim, layer))
            self.mlp.append(get_activation(mlp_activation))
            in_dim = layer
        self.debug_counter = 0
        self.mlp.append(nn.Linear(in_dim, action_space.shape[0]))
        self.mlp.append(nn.Tanh())

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
            x = inputs

        # slice off the first 5 dims for the MLP
        proprio = x[:, :self.mlp_input_size]  # (B,5)
        encoder_ouput = x[:, self.mlp_input_size-1:-1]
        # slice off the next encoder_input_size dims for the encoder
        feat_in = torch.cat([proprio, encoder_ouput], dim=1)

        # pass through the MLP
        for i, layer in enumerate(self.mlp):
            feat_in = layer(feat_in)
        # actions_mean = self.mlp(feat_in)        # (B,2)

        return feat_in, self.log_std_parameter, {}