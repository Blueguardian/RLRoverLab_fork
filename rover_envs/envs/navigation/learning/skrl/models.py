import gymnasium.spaces
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
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
    def __init__(self, in_channels, encoder_features=[16, 32], encoder_activation="leaky_relu"):
        super().__init__()

        self.heightmap_size = torch.sqrt(torch.tensor(in_channels)).int()
        kernel_size = 3
        stride = 1
        padding = 1
        self.encoder_layers = nn.ModuleList()
        in_channels = 1  # 1 channel for heightmap
        for feature in encoder_features:
            self.encoder_layers.append(
                nn.Conv2d(in_channels, feature, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            # self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_layers.append(get_activation(encoder_activation))
            self.encoder_layers.append(nn.Conv2d(feature, feature, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature
        out_channels = in_channels
        flatten_size = [self.heightmap_size, self.heightmap_size]
        for _ in encoder_features:
            w = (flatten_size[0] - kernel_size + 2 * padding) // stride + 1
            h = (flatten_size[1] - kernel_size + 2 * padding) // stride + 1
            w = (w - 2) // 2 + 1
            h = (h - 2) // 2 + 1
            flatten_size = [w, h]

        self.conv_out_features = out_channels * flatten_size[0] * flatten_size[1]
        features = [80, 60]

        self.mlps = nn.ModuleList()
        in_channels = self.conv_out_features
        for feature in features:
            self.mlps.append(nn.Linear(in_channels, feature))
            self.mlps.append(get_activation(encoder_activation))
            in_channels = feature

        self.out_features = features[-1]

    def forward(self, x):
        # x is a flattened heightmap, reshape it to 2D
        x = x.view(-1, 1, self.heightmap_size, self.heightmap_size)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.view(-1, self.conv_out_features)
        for layer in self.mlps:
            x = layer(x)
        return x


class ImageResnet(nn.Module):
    def __init__(self, in_channels, encoder_features=[80, 60], encoder_activation="leaky_relu"):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)

        # Replace input conv if channels ≠ 3 (e.g., grayscale or depth)
        if in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.input_bn = nn.BatchNorm2d(64)
            self.input_relu = nn.ReLU(inplace=True)
            self.input_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.input_conv = base_model.conv1
            self.input_bn = base_model.bn1
            self.input_relu = base_model.relu
            self.input_maxpool = base_model.maxpool

        # Feature extractor (everything up to avgpool)
        self.feature_extractor = nn.Sequential(
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool  # -> [B, 512, 1, 1]
        )

        self.flatten = nn.Flatten()
        self.resnet_out_features = 512

        # Optional MLP projection on top of ResNet
        self.mlps = nn.ModuleList()
        in_features = self.resnet_out_features
        for feature in encoder_features:
            self.mlps.append(nn.Linear(in_features, feature))
            self.mlps.append(get_activation(encoder_activation))
            in_features = feature

        self.out_features = encoder_features[-1]

    def forward(self, x):
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        x = self.input_maxpool(x)

        x = self.feature_extractor(x)
        x = self.flatten(x)  # shape [B, 512]

        for layer in self.mlps:
            x = layer(x)

        return x

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

class GaussianNeuralNetworkConvResnet(GaussianMixin, BaseModel):
    """Gaussian neural network model using ResNet18 for image encoding."""

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
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self,
                               clip_actions=True,
                               clip_log_std=True,
                               min_log_std=-20.0,
                               max_log_std=2.0,
                               reduction="sum"
                               )

        self.image_shape = (3, 112, 112)
        self.depth_shape = (112, 112)
        self.image_flat_size = int(torch.tensor(self.image_shape).prod())
        self.depth_flat_size = int(torch.tensor(self.depth_shape).prod())


        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = mlp_input_size
        self.encoder_rgb = ImageResnet(
            in_channels=3,
            encoder_features=encoder_layers,
            encoder_activation="relu"
        )
        in_channels += self.encoder_rgb.out_features

        self.encoder_depth = ConvHeightmapEncoder(
            in_channels=12544,
            encoder_features=encoder_layers,
            encoder_activation=encoder_activation
        )
        in_channels += self.encoder_depth.out_features

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.mlp = nn.ModuleList()
        for layer in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, layer))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = layer

        self.mlp.append(nn.Linear(in_channels, action_space.shape[0]))
        self.mlp.append(nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(action_space.shape[0]))

    def normalize_image(self, x):

        return (x - self.imagenet_mean) / self.imagenet_std


    def compute(self, states, role="actor"):
        flat = states["states"]  # shape: (B, total_obs_dim)

        # Split flattened input into image and linear_obs
        image_flat = flat[:, :self.image_flat_size]  # (B, 150528)
        depth_flat = flat[:, self.image_flat_size:-self.mlp_input_size]  # (B, depth_size)

        linear_obs = flat[:, -self.mlp_input_size:]
        image = image_flat.view(-1, 3, 112, 112)  # (B, 3, 224, 224)
        # depth_image = depth_flat.view(-1, 224, 224)
        image = self.normalize_image(image)
        image_features = self.encoder_rgb(image)  # (B, 60)
        depth_features = self.encoder_depth(depth_flat)

        # Combine encoded image + linear obs
        x = torch.cat([image_features, depth_features, linear_obs], dim=1)  # (B, 125)

        for i, layer in enumerate(self.mlp):
            x = layer(x)

        return x, self.log_std_parameter, {}

class GaussianNeuralNetworkConv(GaussianMixin, BaseModel):
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
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

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
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = states["states"][:, 0:self.mlp_input_size]
            x = torch.cat([x, encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}


class DeterministicNeuralNetworkConvResnet(DeterministicMixin, BaseModel):
    """Deterministic neural network model using ResNet18 for image encoding."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,  # Assuming [3 proprio + 2 action dims]
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=3,  # Number of image channels
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.image_shape = (3, 112, 112)
        self.image_flat_size = int(torch.tensor(self.image_shape).prod())
        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = mlp_input_size
        self.encoder_rgb = ImageResnet(
            in_channels=3,
            encoder_features=encoder_layers,
            encoder_activation=encoder_activation
        )
        in_channels += self.encoder_rgb.out_features

        self.encoder_depth = ConvHeightmapEncoder(
            in_channels=12544,
            encoder_features=encoder_layers,
            encoder_activation=encoder_activation
        )
        in_channels += self.encoder_depth.out_features

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.mlp = nn.ModuleList()
        for f in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, f))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = f

        self.mlp.append(nn.Linear(in_channels, 1))  # Scalar value output

    def normalize_image(self, x):
        return (x - self.imagenet_mean) / self.imagenet_std

    def compute(self, states, role="value"):
        flat = states["states"]  # shape: (B, total_obs_dim)

        # Split flattened input into image and linear_obs
        image_flat = flat[:, :self.image_flat_size]  # (B, 150528)
        depth_flat = flat[:, self.image_flat_size:-self.mlp_input_size]  # (B, depth_size)
        linear_obs = flat[:, -self.mlp_input_size:]
        image = image_flat.view(-1, 3, 112, 112)  # (B, 3, 224, 224)
        # depth_image = depth_flat.view(-1, 224, 224)

        image = self.normalize_image(image)
        image_features = self.encoder_rgb(image)  # (B, 60)
        depth_features = self.encoder_depth(depth_flat)
        # Combine encoded image + linear obs
        x = torch.cat([image_features, depth_features, linear_obs], dim=1)  # (B, 125)

        for layer in self.mlp:
            x = layer(x)

        return x, {}

class DeterministicNeuralNetworkConv(DeterministicMixin, BaseModel):
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
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

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
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}

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
