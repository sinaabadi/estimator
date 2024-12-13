import numpy as np
import matplotlib.pyplot as plt

# Simulated data for DQN and PPO
np.random.seed(42)  # For reproducibility


def simulate_rewards(model, num_strikes, data_points):
    """
    Simulate average rewards for a given model and number of strikes.

    Parameters:
        model (str): 'DQN' or 'PPO'
        num_strikes (int): 1 or 5
        data_points (ndarray): Array of training data sizes

    Returns:
        rewards (ndarray): Simulated rewards
    """
    if model == 'DQN':
        base = -800
        noise_scale = 100
        convergence_point = 1e6
    elif model == 'PPO':
        base = -600
        noise_scale = 50
        convergence_point = 1e5
    else:
        raise ValueError("Invalid model. Choose 'DQN' or 'PPO'.")

    # Simulate rewards with an asymptotic increase
    rewards = base + 1000 * (1 - np.exp(-data_points / convergence_point))
    rewards += np.random.normal(0, noise_scale, size=data_points.shape)

    # Adjust for multi-strike complexity
    if num_strikes == 5:
        rewards -= 100  # Penalize for multi-strike complexity

    return rewards


# Define training data sizes
data_sizes = np.logspace(3, 6, num=50)

# Simulate rewards for different models and scenarios
rewards_dqn_1_strike = simulate_rewards("DQN", 1, data_sizes)
rewards_dqn_5_strikes = simulate_rewards("DQN", 5, data_sizes)
rewards_ppo_1_strike = simulate_rewards("PPO", 1, data_sizes)
rewards_ppo_5_strikes = simulate_rewards("PPO", 5, data_sizes)

# Plotting
plt.figure(figsize=(12, 6))

# Plot DQN results
plt.plot(data_sizes, rewards_dqn_1_strike, '--', label='DQN (One Strike)')
plt.plot(data_sizes, rewards_dqn_5_strikes, 'x-', label='DQN (Five Strikes)')

# Plot PPO results
plt.plot(data_sizes, rewards_ppo_1_strike, '--', label='PPO (One Strike)')
plt.plot(data_sizes, rewards_ppo_5_strikes, 'x-', label='PPO (Five Strikes)')

# Customize the plot
plt.xscale('log')
plt.xlabel('Training Data Size (# of Data Points)', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.title('Average Reward vs. Training Data Size for DQN and PPO', fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()
