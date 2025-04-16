from gymnasium.spaces.box import Box
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from rover_envs.envs.navigation.learning.skrl.models import (Critic, DeterministicActor, DeterministicNeuralNetwork,
                                                             DeterministicNeuralNetworkConv, GaussianNeuralNetwork,
                                                             GaussianNeuralNetworkConv, GaussianNeuralNetworkConvResnet,
                                                             DeterministicNeuralNetworkConvResnet)


def get_models(agent: str, env: ManagerBasedRLEnv, observation_space: gym.spaces.Space, action_space: Box, conv: bool = False):
    """
    Placeholder function for getting the models.

    Note:
        This function will be further improved in the future, by reading the model config from the experiment config.

    Args:
        agent (str): The agent.

    Returns:
        dict: A dictionary containing the models.
    """

    if agent == "PPO":
        if conv:
            return get_model_gaussian_conv(env, observation_space, action_space)
        return get_model_gaussian(env, observation_space, action_space)
    if agent == "TRPO":
        return get_model_gaussian(env, observation_space, action_space)
    if agent == "RPO":
        return get_model_gaussian(env, observation_space, action_space)
    if agent == "SAC":
        return get_model_double_critic_deterministic(env, observation_space, action_space)
    if agent == "TD3":
        return get_model_double_critic_deterministic(env, observation_space, action_space)

    raise ValueError(f"Agent {agent} not supported.")


def get_model_gaussian(env: ManagerBasedRLEnv, observation_space: Box, action_space: Box):
    models = {}
    encoder_input_size = env.observation_manager.group_obs_term_dim["policy"][-1][0]

    mlp_input_size = 5

    models["policy"] = GaussianNeuralNetwork(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["value"] = DeterministicNeuralNetwork(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    return models


def get_model_gaussian_conv(env: ManagerBasedRLEnv, observation_space: gym.spaces.Space, action_space: Box):
    models = {}

    # Confirm we're working with a Dict space
    if not isinstance(observation_space, gym.spaces.Dict):
        raise ValueError("Expected a Dict observation space for ResNet-based model.")

    # Extract input shapes
    image_shape = observation_space["camera_rgb"].shape      # [3, 240, 320]
    depth_shape = observation_space["camera_depth"].shape
    mlp_input_size = observation_space["linear_obs"].shape[0]+observation_space["actions_taken"].shape[0]   # [N,]

    # mlp_input_size = 3
    encoder_input_channels = image_shape[0]  # Should be 3 for RGB

    models["policy"] = GaussianNeuralNetworkConvResnet(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_channels,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )

    # Optional: implement a ResNet version of your value model later
    models["value"] = DeterministicNeuralNetworkConvResnet(  # Still uses MLP + HeightmapEncoder for now
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_channels,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )

    return models


def get_model_double_critic_deterministic(env: ManagerBasedRLEnv, observation_space: Box, action_space: Box):
    models = {}
    encoder_input_size = env.unwrapped.observation_manager.group_obs_term_dim["policy"][-1][0]

    mlp_input_size = 4

    models["policy"] = DeterministicActor(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["target_policy"] = DeterministicActor(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )

    models["critic_1"] = Critic(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["critic_2"] = Critic(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["target_critic_1"] = Critic(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["target_critic_2"] = Critic(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )

    return models
