from dqn_agent import DQNAgent
from environment import OptionHedgingEnv
from ppo_agent import train_ppo
import matplotlib.pyplot as plt

def train_and_compare():
    env = OptionHedgingEnv(strike_prices=[100])  # Environment instance
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN Training
    dqn_agent = DQNAgent(state_size, action_size)
    episodes = 200
    dqn_rewards = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(env.total_steps):
            action = dqn_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        dqn_agent.replay()
        dqn_rewards.append(total_reward)

    # PPO Training
    ppo_rewards = []
    ppo_env = OptionHedgingEnv(strike_prices=[100])
    ppo_model = train_ppo(ppo_env, total_timesteps=10000)

    for e in range(episodes):
        obs = ppo_env.reset()
        total_reward = 0
        for _ in range(ppo_env.total_steps):
            action, _ = ppo_model.predict(obs)
            obs, reward, done, _ = ppo_env.step(action)
            total_reward += reward
            if done:
                break
        ppo_rewards.append(total_reward)

    # Plot results
    plt.plot(range(episodes), dqn_rewards, label='DQN')
    plt.plot(range(episodes), ppo_rewards, label='PPO')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Exhibit 1: Average Reward vs Training Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_compare()
