import gymnasium.spaces
import torch
import torch.nn as nn
from skrl.models.torch.base import Model as BaseModel
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin


def get_activation(activation_name):
    """Get the activation function by name."""
    activation_fns = {
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "relu6": nn.ReLU6(),
        "selu": nn.SELU(),
    }
    if activation_name not in activation_fns:
        raise ValueError(f"Activation function {activation_name} not supported.")
    return activation_fns[activation_name]


class HeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[80, 60], encoder_activation="leaky_relu"):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class ConvHeightmapEncoder(nn.Module):
    def __init__(self, input_shape, encoder_features=[16, 32], encoder_activation="leaky_relu"):
        """
        Args:
            input_shape (tuple): (C, H, W) — channel-first shape of the image
        """
        super().__init__()
        self.encoder_layers = nn.Sequential()
        in_channels = input_shape[0]

        for feature in encoder_features:
            self.encoder_layers.append(nn.Conv2d(in_channels, feature, kernel_size=3, stride=1, padding=1))
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Compute flattened output size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.encoder_layers(dummy)
            self.conv_out_features = conv_out.view(1, -1).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(self.conv_out_features, 80),
            get_activation(encoder_activation),
            nn.Linear(80, 60),
            get_activation(encoder_activation)
        )

        self.out_features = 60

    def forward(self, x):
        # Expect input as (B, C, H, W)
        x = self.encoder_layers(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

# class ConvHeightmapEncoder(nn.Module):
#     def __init__(self, in_channels, encoder_features=[16, 32], encoder_activation="leaky_relu"):
#         super().__init__()
#         self.heightmap_size = torch.sqrt(torch.tensor(in_channels)).int()
#         kernel_size = 3
#         stride = 1
#         padding = 1
#         self.encoder_layers = nn.ModuleList()
#         in_channels = 1  # 1 channel for heightmap
#         for feature in encoder_features:
#             self.encoder_layers.append(
#                 nn.Conv2d(in_channels, feature, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
#             self.encoder_layers.append(nn.BatchNorm2d(feature))
#             # self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#             self.encoder_layers.append(get_activation(encoder_activation))
#             self.encoder_layers.append(nn.Conv2d(feature, feature, kernel_size=kernel_size,
#                                        stride=stride, padding=padding, bias=False))
#             self.encoder_layers.append(nn.BatchNorm2d(feature))
#             self.encoder_layers.append(get_activation(encoder_activation))
#             self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#             in_channels = feature
#         out_channels = in_channels
#         flatten_size = [self.heightmap_size, self.heightmap_size]
#         for _ in encoder_features:
#             w = (flatten_size[0] - kernel_size + 2 * padding) // stride + 1
#             h = (flatten_size[1] - kernel_size + 2 * padding) // stride + 1
#             w = (w - 2) // 2 + 1
#             h = (h - 2) // 2 + 1
#             flatten_size = [w, h]
#
#         self.conv_out_features = out_channels * flatten_size[0] * flatten_size[1]
#         features = [80, 60]
#
#         self.mlps = nn.ModuleList()
#         in_channels = self.conv_out_features
#         for feature in features:
#             self.mlps.append(nn.Linear(in_channels, feature))
#             self.mlps.append(get_activation(encoder_activation))
#             in_channels = feature
#
#         self.out_features = features[-1]
#
#     def forward(self, x):
#         # x is a flattened heightmap, reshape it to 2D
#         x = x.view(-1, 1, self.heightmap_size, self.heightmap_size)
#         for layer in self.encoder_layers:
#             x = layer(x)
#
#         x = x.view(-1, self.conv_out_features)
#         for layer in self.mlps:
#             x = layer(x)
#         return x


class GaussianNeuralNetwork(GaussianMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        # Split the states into proprioception and heightmap if the heightmap is used.
        if self.encoder_input_size is None:
            x = states["states"]
        else:
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = states["states"][:, 0:self.mlp_input_size]
            x = torch.cat([x, encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}


class DeterministicNeuralNetwork(DeterministicMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}


class DeterministicActor(DeterministicMixin, BaseModel):
    """Deterministic actor model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the deterministic actor model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, action_space))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}


class Critic(DeterministicMixin, BaseModel):
    """Critic model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the critic model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = torch.cat([states["states"], states["taken_actions"]], dim=1)
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)


        return x, {}

class GaussianNeuralNetworkConv(GaussianMixin, BaseModel):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.device = device
        self.image_encoders = nn.ModuleDict()
        self.scalar_keys = []
        scalar_input_size = 0

        # Dynamically inspect and separate image vs scalar terms
        if isinstance(observation_space, gymnasium.spaces.Dict):
            for name, space in observation_space.spaces.items():
                shape = space.shape
                is_image = False
                if len(shape) == 3 and shape[-1] in [1, 3] and min(shape[:2]) >= 32:
                    is_image = True
                elif len(shape) == 2 and min(shape) >= 32:
                    is_image = True

                if is_image:
                    channels = shape[2] if len(shape) == 3 else 1
                    height = shape[0]
                    width = shape[1]

                    conv = nn.Sequential(
                        nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                        nn.Flatten()
                    )
                    with torch.no_grad():
                        dummy = torch.zeros(1, channels, height, width)
                        conv_out_size = conv(dummy).view(1, -1).shape[1]
                    encoder = nn.Sequential(conv, nn.Linear(conv_out_size, 128), nn.ReLU())
                    self.image_encoders[name] = encoder
                else:
                    self.scalar_keys.append(name)
                    scalar_input_size += int(torch.tensor(shape).prod().item())

        # Scalar MLP encoder
        self.scalar_encoder = None
        if scalar_input_size > 0:
            self.scalar_encoder = nn.Sequential(
                nn.Linear(scalar_input_size, 128), get_activation(mlp_activation),
                nn.Linear(128, 128), get_activation(mlp_activation)
            )

        # Final MLP head
        fusion_size = 128 * len(self.image_encoders)
        if self.scalar_encoder:
            fusion_size += 128

        self.mean_layer = nn.Linear(fusion_size, action_space.shape[0])
        self.log_std_layer = nn.Linear(fusion_size, action_space.shape[0])

    def compute(self, obs, role="actor"):
        img_feats = []
        scalar_feats = []

        for name, val in obs.items():
            if name in self.image_encoders:
                x = val
                if x.ndim == 4 and x.shape[-1] in [1, 3]:
                    x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                elif x.ndim == 3 and x.shape[-1] in [1, 3]:
                    x = x.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
                elif x.ndim == 3:  # (B, H, W) grayscale
                    x = x.unsqueeze(1)  # (B, 1, H, W)
                feat = self.image_encoders[name](x)
                img_feats.append(feat)
            elif name in self.scalar_keys:
                scalar_feats.append(val.view(val.shape[0], -1))

        scalar_out = None
        if scalar_feats:
            scalar_input = torch.cat(scalar_feats, dim=-1)
            scalar_out = self.scalar_encoder(scalar_input) if self.scalar_encoder else scalar_input

        feature_parts = []
        if img_feats:
            feature_parts.extend(img_feats)
        if scalar_out is not None:
            feature_parts.append(scalar_out)

        if not feature_parts:
            batch_size = list(obs.values())[0].shape[0]
            dummy_output = torch.zeros(batch_size, self.mean_layer.out_features, device=self.device)
            return dummy_output, torch.zeros_like(dummy_output), {}

        fused = torch.cat(feature_parts, dim=-1)
        mean = self.mean_layer(fused)
        log_std = self.log_std_layer(fused)
        return mean, log_std, {}


# class GaussianNeuralNetworkConv(GaussianMixin, BaseModel):
#     def __init__(
#                         self,
#                         observation_space,
#                         action_space,
#                         device,
#                         mlp_input_size=5,
#                         mlp_layers=[256, 160, 128],
#                         mlp_activation="leaky_relu",
#                         encoder_input_size=None,
#                         encoder_layers=[80, 60],
#                         encoder_activation="leaky_relu",
#                         **kwargs,
#                     ):
#         BaseModel.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(
#             self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
#         )
#
#         # Existing init code ...
#         # NEW: Identify image observations and define encoders
#         self.image_encoders = nn.ModuleDict()
#         scalar_input_size = 0
#         # Assume observation_space is a Dict from gym.spaces (because we set concatenate_terms=False)
#         for name, space in observation_space.spaces.items():
#             if len(space.shape) == 3:
#                 # Image observation detected
#                 channels = space.shape[0] if space.shape[0] in [1,3] else space.shape[2]
#                 height  = space.shape[1] if space.shape[0] in [1,3] else space.shape[0]
#                 width   = space.shape[2] if space.shape[0] in [1,3] else space.shape[1]
#                 # Define conv encoder for this image
#                 conv_layers = nn.Sequential(
#                     nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.ReLU(),
#                     nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
#                     nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
#                     nn.Flatten()
#                 )
#                 # Determine flattened conv output size:
#                 with torch.no_grad():
#                     dummy = torch.zeros(1, channels, height, width)
#                     conv_out_len = conv_layers(dummy).shape[1]
#                 encoder = nn.Sequential(conv_layers, nn.Linear(conv_out_len, 128), nn.ReLU())
#                 self.image_encoders[name] = encoder
#             else:
#                 # Scalar observation
#                 scalar_input_size += space.shape[0]
#         # Define MLP for scalar inputs if any
#         self.scalar_encoder = None
#         if scalar_input_size > 0:
#             self.scalar_encoder = nn.Sequential(
#                 nn.Linear(scalar_input_size, 128), nn.ReLU(),
#                 nn.Linear(128, 128), nn.ReLU()
#             )
#         # Define layers for output (Gaussian parameters)
#         total_feat_len = 128 * len(self.image_encoders) + (128 if self.scalar_encoder else 0)
#         self.mean_layer = nn.Linear(total_feat_len, action_space.shape[0])
#         self.log_std_layer = nn.Linear(total_feat_len, action_space.shape[0])
#     def compute(self, obs, role="actor"):
#         # Expect obs as dict of tensors
#         img_feats = []
#         scalar_list = []
#         for name, val in obs.items():
#             if name in self.image_encoders:
#                 # Ensure channel-first
#                 x = val
#                 if x.dim() == 4 and x.shape[-1] in [1,3]:
#                     x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#                 elif x.dim() == 3 and x.shape[-1] in [1,3]:
#                     x = x.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
#                 feat = self.image_encoders[name](x)
#                 img_feats.append(feat)
#             else:
#                 scalar_list.append(val if val.dim()>1 else val.unsqueeze(0))  # ensure batch dim
#         scalar_feat = None
#         if scalar_list:
#             scalar_input = torch.cat(scalar_list, dim=-1)
#             scalar_feat = self.scalar_encoder(scalar_input) if self.scalar_encoder else scalar_input
#         # concatenate all features
#         fused = torch.cat([*img_feats, scalar_feat] if scalar_feat is not None else img_feats, dim=-1)
#         # Compute mean and log_std for Gaussian policy
#         mu = self.mean_layer(fused)
#         log_sigma = self.log_std_layer(fused)
#         return mu, log_sigma


# class GaussianNeuralNetworkConv(GaussianMixin, BaseModel):
#     """Gaussian neural network model."""
#
#     def __init__(
#         self,
#         observation_space,
#         action_space,
#         device,
#         mlp_input_size=5,
#         mlp_layers=[256, 160, 128],
#         mlp_activation="leaky_relu",
#         encoder_input_size=None,
#         encoder_layers=[80, 60],
#         encoder_activation="leaky_relu",
#         **kwargs,
#     ):
#         """Initialize the Gaussian neural network model.
#
#         Args:
#             observation_space (gym.spaces.Space): The observation space of the environment.
#             action_space (gym.spaces.Space): The action space of the environment.
#             device (torch.device): The device to use for computation.
#             encoder_features (list): The number of features for each encoder layer.
#             encoder_activation (str): The activation function to use for each encoder layer.
#         """
#         BaseModel.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(
#             self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
#         )
#
#         self.mlp_input_size = mlp_input_size
#         self.encoder_input_size = encoder_input_size
#
#         in_channels = self.mlp_input_size
#         if self.encoder_input_size is not None:
#             self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
#             in_channels += self.encoder.out_features
#
#         self.mlp = nn.ModuleList()
#
#         for feature in mlp_layers:
#             self.mlp.append(nn.Linear(in_channels, feature))
#             self.mlp.append(get_activation(mlp_activation))
#             in_channels = feature
#
#         action_space = action_space.shape[0]
#         self.mlp.append(nn.Linear(in_channels, action_space))
#         self.mlp.append(nn.Tanh())
#         self.log_std_parameter = nn.Parameter(torch.zeros(action_space))
#
#     def compute(self, states, role="actor"):
#         # Split the states into proprioception and heightmap if the heightmap is used.
#         if self.encoder_input_size is None:
#             x = states["states"]
#         else:
#             encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
#             x = states["states"][:, 0:self.mlp_input_size]
#             x = torch.cat([x, encoder_output], dim=1)
#
#         # Compute the output of the MLP.
#         for layer in self.mlp:
#             x = layer(x)
#
#         return x, self.log_std_parameter, {}


class DeterministicNeuralNetworkConv(DeterministicMixin, BaseModel):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.device = device
        self.image_encoders = nn.ModuleDict()
        self.scalar_keys = []
        scalar_input_size = 0

        if isinstance(observation_space, gymnasium.spaces.Dict):
            for name, space in observation_space.spaces.items():
                shape = space.shape
                is_image = False
                if len(shape) == 3 and shape[-1] in [1, 3] and min(shape[:2]) >= 32:
                    is_image = True
                elif len(shape) == 2 and min(shape) >= 32:
                    is_image = True

                if is_image:
                    channels = shape[2] if len(shape) == 3 else 1
                    height = shape[0]
                    width = shape[1]

                    conv = nn.Sequential(
                        nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                        nn.Flatten()
                    )

                    with torch.no_grad():
                        dummy = torch.zeros(1, channels, height, width)
                        conv_out_size = conv(dummy).view(1, -1).shape[1]
                    encoder = nn.Sequential(conv, nn.Linear(conv_out_size, 128), nn.ReLU())
                    print(encoder)
                    self.image_encoders[name] = encoder
                else:
                    self.scalar_keys.append(name)
                    scalar_input_size += int(torch.tensor(shape).prod().item())

        self.scalar_encoder = None
        if scalar_input_size > 0:
            self.scalar_encoder = nn.Sequential(
                nn.Linear(scalar_input_size, 128), get_activation(mlp_activation),
                nn.Linear(128, 128), get_activation(mlp_activation)
            )

        fusion_size = 128 * len(self.image_encoders)
        if self.scalar_encoder:
            fusion_size += 128

        self.value_layer = nn.Linear(fusion_size, 1)

    def compute(self, obs, role="critic"):
        if "states" in obs and isinstance(obs["states"], torch.Tensor):
            x = obs["states"]
            B = x.shape[0]

            # Define known sizes
            rgb_size = 3 * 240 * 320
            depth_size = 1 * 240 * 320
            scalar_size = 2  # adjust if needed

            # Slice
            offset = 0
            camera_rgb = x[:, offset:offset + rgb_size].view(B, 240, 320, 3)
            offset += rgb_size
            camera_depth = x[:, offset:offset + depth_size].view(B, 240, 320, 1)
            offset += depth_size
            linear_obs = x[:, offset:offset + scalar_size]

            # Reconstruct dictionary
            obs = {
                "camera_rgb": camera_rgb,
                "camera_depth": camera_depth,
                "linear_obs": linear_obs
            }

        img_feats = []
        scalar_feats = []

        # if isinstance(obs, dict) and "states" in obs:
        #     if not obs["states"].requires_grad:
        #         obs["states"] = obs["states"].detach().clone().requires_grad_()

        for name, val in obs.items():
            if name in self.image_encoders:
                x = val
                if x.ndim == 4 and x.shape[-1] in [1, 3]:
                    x = x.permute(0, 3, 1, 2)
                elif x.ndim == 3 and x.shape[-1] in [1, 3]:
                    x = x.permute(2, 0, 1).unsqueeze(0)
                elif x.ndim == 3:
                    x = x.unsqueeze(1)
                feat = self.image_encoders[name](x)

                img_feats.append(feat)
            elif name in self.scalar_keys:
                scalar_feats.append(val.view(val.shape[0], -1))
        scalar_out = None
        if scalar_feats:

            scalar_input = torch.cat(scalar_feats, dim=-1)
            scalar_out = self.scalar_encoder(scalar_input) if self.scalar_encoder else scalar_input

        feature_parts = []
        if img_feats:
            feature_parts.extend(img_feats)
        if scalar_out is not None:
            feature_parts.append(scalar_out)
        if not feature_parts:
            print("No features found!")
            batch_size = list(obs.values())[0].shape[0]
            dummy_output = torch.zeros(batch_size, 1, device=self.device)
            return dummy_output, {}
        fused = torch.cat(feature_parts, dim=-1)
        value = self.value_layer(fused)
        return value, {}

# class DeterministicNeuralNetworkConv(DeterministicMixin, BaseModel):
#     """Gaussian neural network model."""
#
#     def __init__(
#         self,
#         observation_space,
#         action_space,
#         device,
#         mlp_input_size=4,
#         mlp_layers=[256, 160, 128],
#         mlp_activation="leaky_relu",
#         encoder_input_size=None,
#         encoder_layers=[80, 60],
#         encoder_activation="leaky_relu",
#         **kwargs,
#     ):
#         """Initialize the Gaussian neural network model.
#
#         Args:
#             observation_space (gym.spaces.Space): The observation space of the environment.
#             action_space (gym.spaces.Space): The action space of the environment.
#             device (torch.device): The device to use for computation.
#             encoder_features (list): The number of features for each encoder layer.
#             encoder_activation (str): The activation function to use for each encoder layer.
#         """
#         BaseModel.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions=False)
#
#         self.mlp_input_size = mlp_input_size
#         self.encoder_input_size = encoder_input_size
#
#         in_channels = self.mlp_input_size
#         if self.encoder_input_size is not None:
#             self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
#             in_channels += self.encoder.out_features
#
#         self.mlp = nn.ModuleList()
#
#         action_space = action_space.shape[0]
#         for feature in mlp_layers:
#             self.mlp.append(nn.Linear(in_channels, feature))
#             self.mlp.append(get_activation(mlp_activation))
#             in_channels = feature
#
#         self.mlp.append(nn.Linear(in_channels, 1))
#
#     def compute(self, states, role="actor"):
#         if self.encoder_input_size is None:
#             x = states["states"]
#         else:
#             x = states["states"][:, :self.mlp_input_size]
#             encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
#             x = torch.cat([x, encoder_output], dim=1)
#
#         for layer in self.mlp:
#             x = layer(x)
#
#         return x, {}
