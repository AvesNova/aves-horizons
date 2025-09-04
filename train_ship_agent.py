import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium import spaces

from gym_env.ship_env import ShipEnv


class ShipTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class ShipTransformerPolicy(ActorCriticPolicy):
    """Custom transformer-based policy for the ship environment."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_ships=8,
        obs_per_ship=10,
        d_model=64,
        nhead=4,
        num_layers=3,
        **kwargs,  # Accept additional kwargs from PPO
    ):
        # Store transformer parameters
        self.n_ships = n_ships
        self.obs_per_ship = obs_per_ship
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Calculate transformer output size
        self.transformer_output_size = self.d_model * self.n_ships

        # Create the network architecture for the parent class
        # The input size to the first layer will be set correctly by overriding _build_mlp_extractor
        net_arch = {
            "pi": [256, 128, 64],  # Policy network hidden layers
            "vf": [256, 128, 64],  # Value network hidden layers
        }

        # Filter out kwargs that we don't pass to parent
        parent_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "optimizer_class",
                "optimizer_kwargs",
                "features_extractor_class",
                "features_extractor_kwargs",
                "normalize_images",
                "optimize_memory_usage",
            ]
        }

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=nn.ReLU,
            normalize_images=False,
            **parent_kwargs,
        )

        # Create transformer components after parent initialization
        self.embedding = nn.Linear(self.obs_per_ship, self.d_model)
        self.transformer = nn.ModuleList(
            [
                ShipTransformerBlock(self.d_model, self.nhead)
                for _ in range(self.num_layers)
            ]
        )

    def _build_mlp_extractor(self) -> None:
        """Build MLP networks with correct input size for transformer output."""
        from stable_baselines3.common.torch_layers import MlpExtractor

        # Build MLP with transformer output size as input
        self.mlp_extractor = MlpExtractor(
            feature_dim=self.transformer_output_size,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features through transformer before MLP networks."""
        # Calculate expected total observation size
        expected_size = self.n_ships * self.obs_per_ship

        # Handle both single and batch observations
        if len(obs.shape) == 1:
            # Single observation
            if obs.shape[0] != expected_size:
                raise ValueError(
                    f"Expected observation size {expected_size}, got {obs.shape[0]}"
                )
            obs = obs.view(1, self.n_ships, self.obs_per_ship)
        else:
            # Batch of observations
            if obs.shape[1] != expected_size:
                raise ValueError(
                    f"Expected observation size {expected_size}, got {obs.shape[1]}"
                )
            obs = obs.view(-1, self.n_ships, self.obs_per_ship)

        # Embed each ship's state
        x = self.embedding(obs)

        # Pass through transformer layers
        for transformer in self.transformer:
            x = transformer(x)

        # Flatten for MLP heads
        return x.reshape(obs.shape[0], -1)

    def get_distribution(self, obs: torch.Tensor):
        """Get action distribution using transformer feature extraction."""
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values using transformer feature extraction."""
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        """Evaluate actions using transformer feature extraction."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """Forward pass through transformer policy."""
        # Extract features using transformer
        features = self.extract_features(obs)

        # Pass through MLP extractor to get latent representations
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get distribution and values from the latent representations
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)

        # Sample actions from distribution
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob


def make_env(n_ships=8, render_mode=None):
    """Create ship environment with given parameters."""
    return lambda: ShipEnv(n_ships=n_ships, render_mode=render_mode)


def train_ship_agent(total_timesteps=1000000, n_ships=8, n_envs=8):
    """Train a ship agent using PPO with transformer policy."""
    from stable_baselines3.common.monitor import Monitor

    # Create vectorized environment
    env = make_env(n_ships=n_ships)()

    # Initialize PPO agent with custom policy
    model = PPO(
        policy=ShipTransformerPolicy,
        env=env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            n_ships=n_ships,
            obs_per_ship=10,
            d_model=64,
            nhead=4,
            num_layers=3,
        ),
        verbose=1,
    )

    # Set up callbacks
    # Wrap evaluation environment with Monitor to avoid warning
    eval_env = Monitor(make_env(n_ships=n_ships)())
    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=10,
        eval_freq=10000,
        log_path="./logs/",
        best_model_save_path="./best_model/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, save_path="./checkpoints/", name_prefix="ship_model"
    )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    model.save("ship_final_model")

    return model


def evaluate_and_render(model_path="ship_final_model", n_episodes=5):
    """Evaluate and render a trained model."""
    # Load model
    model = PPO.load(model_path)

    # Create environment with rendering
    env = make_env(render_mode="human")()

    # Run evaluation episodes
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    # Train agent
    model = train_ship_agent(total_timesteps=1000000)

    # Evaluate and visualize
    evaluate_and_render()
