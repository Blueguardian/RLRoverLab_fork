import gymnasium.spaces
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from skrl.models.torch.base import Model as BaseModel
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin
from torchvision.transforms.functional import to_pil_image
import os
import numpy as np



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
        x#  = x.view(-1, 1, self.heightmap_size, self.heightmap_size)
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
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        # Extract space sizes from observation_space
        self.obs_sizes = {}
        for key, space in observation_space.items():
            size = 1
            for dim in space.shape:
                size *= dim
            self.obs_sizes[key] = size

        self.mlp_input_size = self.obs_sizes.get("distance", 0) + self.obs_sizes.get("heading", 0) + self.obs_sizes.get(
            "relative_goal_orientation", 0) + self.obs_sizes.get("actions_taken", 0)
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.obs_sizes.get("camera_rgb", None):
            self.encoder_rgb = ImageResnet(
                in_channels=3,
                encoder_features=encoder_layers,
                encoder_activation="relu"
            )
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            in_channels += self.encoder_rgb.out_features

        if self.obs_sizes.get("camera_depth", None):
            self.encoder_depth = ConvHeightmapEncoder(
                in_channels=self.obs_sizes["camera_depth"],
                encoder_features=encoder_layers,
                encoder_activation=encoder_activation
            )
            in_channels += self.encoder_depth.out_features

        if self.obs_sizes.get("raycaster_cam", None):
            self.encoder_depthCam = ConvHeightmapEncoder(
                in_channels=self.obs_sizes["raycaster_cam"],
                encoder_features=encoder_layers,
                encoder_activation=encoder_activation
            )
            in_channels += self.encoder_depthCam.out_features

        if self.obs_sizes.get("raycaster", None):
            self.encoder_raycaster = ConvHeightmapEncoder(
                in_channels=self.obs_sizes["raycaster"],
                encoder_features=encoder_layers,
                encoder_activation=encoder_activation
            )
            in_channels += self.encoder_raycaster.out_features

        self.mlp = nn.ModuleList()
        for layer in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, layer))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = layer
        self.debug_counter = 0
        self.mlp.append(nn.Linear(in_channels, action_space.shape[0]))
        self.mlp.append(nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(action_space.shape[0]))

    def normalize_image(self, x):
        return ((x.float()/255) - self.imagenet_mean) / self.imagenet_std

    def save_tensor_to_csv(self, tensor, name="tensor", out_dir="/workspace/isaac_rover/debug_tensors", env_idx=0, step=0):
        os.makedirs(out_dir, exist_ok=True)

        # Convert to CPU + NumPy
        tensor = tensor.detach().cpu()
        np_tensor = tensor.numpy()

        # Save flattened version
        flat_path = os.path.join(out_dir, f"{name}_env{env_idx}_step{step}_flat.csv")
        np.savetxt(flat_path, np_tensor.flatten(), delimiter=",", fmt="%.6f")

        # If it's 2D or 3D (like depth or image channels), save shape-preserving CSV
        if np_tensor.ndim == 2:
            shape_path = os.path.join(out_dir, f"{name}_env{env_idx}_step{step}_matrix.csv")
            np.savetxt(shape_path, np_tensor, delimiter=",", fmt="%.6f")
        elif np_tensor.ndim == 3:
            for i, channel in enumerate(np_tensor):
                ch_path = os.path.join(out_dir, f"{name}_env{env_idx}_step{step}_ch{i}.csv")
                np.savetxt(ch_path, channel, delimiter=",", fmt="%.6f")

    def save_rgb_depth(self, rgb_tensor, depth_tensor, step_env=0, env_idx=0, out_dir="/workspace/isaac_rover/debug_images"):
        os.makedirs(out_dir, exist_ok=True)

        # ---- RGB ----
        print(f"RGB stats — min: {rgb_tensor.min()}, max: {rgb_tensor.max()}, dtype: {rgb_tensor.dtype}")
        rgb_tensor = rgb_tensor[env_idx] / 255
        # rgb_tensor = rgb_tensor[env_idx].to(torch.uint8)
        rgb = rgb_tensor.detach().to("cpu")
        rgb_image = to_pil_image(rgb)
        rgb_image.save(f"{out_dir}/rgb_step{step_env}_env{env_idx}.png")

        # ---- Depth ----
        print(f"RGB stats — min: {depth_tensor.min()}, max: {depth_tensor.max()}, dtype: {depth_tensor.dtype}")
        depth = depth_tensor[env_idx].detach().to("cpu").squeeze(0)  # Remove channel if 1xHxW
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)  # Normalize to 0-1
        depth_image = to_pil_image(depth)
        depth_image.save(f"{out_dir}/depth_step{step_env}_env{env_idx}.png")

    def compute(self, states, role="actor"):
        # Reform the observation space TODO: Check for correct order with rgb
        obs_space = self.tensor_to_space(states["states"], self.observation_space)

        # Extract the observations
        depth = obs_space["camera_depth"]
        # depth = obs_space["camera_depth"]
        proprioceptive_obs = torch.cat([obs_space["distance"], obs_space["heading"], obs_space["relative_goal_orientation"], obs_space["actions_taken"]], dim=1)
        rgb = obs_space["camera_rgb"]
        rgb_normrect = self.normalize_image(rgb)
        rgb_features = self.encoder_rgb(rgb_normrect)  # (B, 60)
        # depth_features = self.encoder_depth(depth.flatten(start_dim=1))
        depth_features = self.encoder_depth(depth)

        # DEBUG output image and depth image (greyscale)
        if self.debug_counter % 500 == 0:
            self.save_rgb_depth(rgb, depth, step_env=self.debug_counter)
            # self.save_tensor_to_csv(rgb_normrect, "rgb_normrect", step=self.debug_counter)
            # self.save_tensor_to_csv(rgb_features, "rgb_features", step=self.debug_counter)
            # self.save_tensor_to_csv(depth_features, "depth_features", step=self.debug_counter)
        self.debug_counter += 1

        x = torch.cat([rgb_features, depth_features, proprioceptive_obs], dim=1)
        # x = torch.cat([depth_features, proprioceptive_obs], dim=1)
        for i, layer in enumerate(self.mlp):
            x = layer(x)

        return x, self.log_std_parameter, {}

class DeterministicNeuralNetworkConvResnet(DeterministicMixin, BaseModel):
    """Deterministic neural network model using ResNet18 for image encoding."""

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
        DeterministicMixin.__init__(self,
                               clip_actions=False
                               )
        # Extract space sizes from observation_space
        self.obs_sizes = {}
        for key, space in observation_space.items():
            size = 1
            for dim in space.shape:
                size *= dim
            self.obs_sizes[key] = size

        self.mlp_input_size = self.obs_sizes.get("distance", 0)+self.obs_sizes.get("heading",0)+self.obs_sizes.get("relative_goal_orientation", 0)+self.obs_sizes.get("actions_taken",0)
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.obs_sizes.get("camera_rgb", None):
            self.encoder_rgb = ImageResnet(
                in_channels=3,
                encoder_features=encoder_layers,
                encoder_activation="relu"
            )
            in_channels += self.encoder_rgb.out_features

        if self.obs_sizes.get("camera_depth", None):
            self.encoder_depth = ConvHeightmapEncoder(
                in_channels=self.obs_sizes["camera_depth"],
                encoder_features=encoder_layers,
                encoder_activation=encoder_activation
            )
            in_channels += self.encoder_depth.out_features

        if self.obs_sizes.get("raycaster_cam", None):
            self.encoder_depthCam = ConvHeightmapEncoder(
                in_channels=self.obs_sizes["raycaster_cam"],
                encoder_features=encoder_layers,
                encoder_activation=encoder_activation
            )
            in_channels += self.encoder_depthCam.out_features

        if self.obs_sizes.get("raycaster", None):
            self.encoder_raycaster = ConvHeightmapEncoder(
                in_channels=self.obs_sizes["raycaster"],
                encoder_features=encoder_layers,
                encoder_activation=encoder_activation
            )
            in_channels += self.encoder_raycaster.out_features

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.mlp = nn.ModuleList()
        for layer in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, layer))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = layer
        self.debug_counter = 0
        self.mlp.append(nn.Linear(in_channels, 1))

    def normalize_image(self, x):
        return ((x.float() / 255) - self.imagenet_mean) / self.imagenet_std

    def compute(self, states, role="value"):
        # Reform the observation space TODO: Check for correct order with rgb
        obs_space = self.tensor_to_space(states["states"], self.observation_space)

        # Extract the observations
        depth = obs_space["camera_depth"]
        # depth = obs_space["camera_depth"]
        proprioceptive_obs = torch.cat(
            [obs_space["distance"], obs_space["heading"], obs_space["relative_goal_orientation"], obs_space["actions_taken"]],
            dim=1)
        rgb = obs_space["camera_rgb"]
        rgb_normrect = self.normalize_image(rgb)
        rgb_features = self.encoder_rgb(rgb_normrect)  # (B, 60)
        # depth_features = self.encoder_depth(depth.flatten(start_dim=1))
        depth_features = self.encoder_depth(depth)

        # DEBUG output image and depth image (greyscale)
        # if self.debug_counter % 500 == 0:
        #     self.save_rgb_depth(rgb, depth, step_env=self.debug_counter)
        #     self.save_tensor_to_csv(rgb_normrect, "rgb_normrect", step=self.debug_counter)
        #     self.save_tensor_to_csv(rgb_features, "rgb_features", step=self.debug_counter)
        #     self.save_tensor_to_csv(depth_features, "depth_features", step=self.debug_counter)
        # self.debug_counter += 1

        x = torch.cat([rgb_features, depth_features, proprioceptive_obs], dim=1)
        # x = torch.cat([depth_features, proprioceptive_obs], dim=1)
        for i, layer in enumerate(self.mlp):
            x = layer(x)

        return x, {}