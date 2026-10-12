"""
Multi-Head Policy Network (rl/policy_network.py)
Multi-head Actor-Critic neural network outputting joint Guidance and EW action distributions.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Module:
        pass
    nn = type('nn', (), {'Module': Module})


class MultiHeadPolicyNet(nn.Module):
    """
    Joint multi-head network:
    - Shared Backbone: Evaluates situational feature representations.
    - Guidance Actor Head: Continuous normal distributions for [AoA, Bank angle].
    - EW Actor Head: Categorical distribution for [Mode] & Continuous for [Power level].
    - Critic Head: State value estimation V(s).
    """

    def __init__(self, state_dim: int = 16, hidden_dims: list = [256, 256, 128]):
        if HAS_TORCH:
            super().__init__()

            # Shared feature extractor
            layers = []
            in_dim = state_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.ReLU())
                in_dim = h_dim
            self.backbone = nn.Sequential(*layers)

            # Guidance Action Head (Continuous [alpha, sigma])
            self.guidance_mean = nn.Linear(in_dim, 2)
            self.guidance_log_std = nn.Parameter(torch.zeros(1, 2))

            # EW Action Head (Discrete Mode [4] + Continuous Power [1])
            self.ew_mode_logits = nn.Linear(in_dim, 4)
            self.ew_power_mean = nn.Linear(in_dim, 1)
            self.ew_power_log_std = nn.Parameter(torch.zeros(1, 1))

            # Critic Head V(s)
            self.critic = nn.Linear(in_dim, 1)

    def forward(self, state):
        if not HAS_TORCH:
            return None, None
        features = self.backbone(state)
        value = self.critic(features)
        return features, value

    def sample_action(self, state, deterministic: bool = False):
        if not HAS_TORCH:
            return None, None, None
        features, value = self.forward(state)

        # Guidance continuous sampling
        g_mean = torch.tanh(self.guidance_mean(features))
        g_std = torch.exp(self.guidance_log_std).expand_as(g_mean)
        g_dist = Normal(g_mean, g_std)

        # EW mode discrete sampling
        ew_logits = self.ew_mode_logits(features)
        ew_mode_dist = Categorical(logits=ew_logits)

        # EW power continuous sampling
        ew_p_mean = torch.sigmoid(self.ew_power_mean(features))
        ew_p_std = torch.exp(self.ew_power_log_std).expand_as(ew_p_mean)
        ew_p_dist = Normal(ew_p_mean, ew_p_std)

        if deterministic:
            g_action = g_mean
            ew_mode_action = torch.argmax(ew_logits, dim=-1)
            ew_p_action = ew_p_mean
        else:
            g_action = g_dist.sample()
            ew_mode_action = ew_mode_dist.sample()
            ew_p_action = ew_p_dist.sample()

        # Combine actions into single vector
        action = torch.cat([g_action, ew_mode_action.unsqueeze(-1).float(), ew_p_action], dim=-1)

        # Compute log probabilities
        log_prob_g = g_dist.log_prob(g_action).sum(dim=-1, keepdim=True)
        log_prob_mode = ew_mode_dist.log_prob(ew_mode_action).unsqueeze(-1)
        log_prob_power = ew_p_dist.log_prob(ew_p_action)

        total_log_prob = log_prob_g + log_prob_mode + log_prob_power

        return action, total_log_prob, value

    def evaluate_actions(self, state, action):
        if not HAS_TORCH:
            return None, None, None
        features, value = self.forward(state)

        g_action = action[:, 0:2]
        ew_mode_action = action[:, 2].long()
        ew_p_action = action[:, 3:4]

        # Guidance log prob
        g_mean = torch.tanh(self.guidance_mean(features))
        g_std = torch.exp(self.guidance_log_std).expand_as(g_mean)
        g_dist = Normal(g_mean, g_std)
        log_prob_g = g_dist.log_prob(g_action).sum(dim=-1, keepdim=True)
        entropy_g = g_dist.entropy().sum(dim=-1, keepdim=True)

        # EW mode log prob
        ew_logits = self.ew_mode_logits(features)
        ew_mode_dist = Categorical(logits=ew_logits)
        log_prob_mode = ew_mode_dist.log_prob(ew_mode_action).unsqueeze(-1)
        entropy_mode = ew_mode_dist.entropy().unsqueeze(-1)

        # EW power log prob
        ew_p_mean = torch.sigmoid(self.ew_power_mean(features))
        ew_p_std = torch.exp(self.ew_power_log_std).expand_as(ew_p_mean)
        ew_p_dist = Normal(ew_p_mean, ew_p_std)
        log_prob_power = ew_p_dist.log_prob(ew_p_action)
        entropy_power = ew_p_dist.entropy()

        total_log_prob = log_prob_g + log_prob_mode + log_prob_power
        total_entropy = entropy_g + entropy_mode + entropy_power

        return value, total_log_prob, total_entropy
