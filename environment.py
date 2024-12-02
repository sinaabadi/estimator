import numpy as np
import gym
from gym import spaces

class OptionHedgingEnv(gym.Env):
    def __init__(self, strike_prices=[100], cost_multiplier=0.01, days=10, steps_per_day=5):
        super(OptionHedgingEnv, self).__init__()
        self.strike_prices = strike_prices
        self.cost_multiplier = cost_multiplier
        self.steps_per_day = steps_per_day
        self.days = days
        self.total_steps = self.days * self.steps_per_day
        self.np_random = None  # Add this for setting random seed
        self.reset()

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, min(self.strike_prices)]),
            high=np.array([np.inf, self.total_steps, np.inf, max(self.strike_prices)]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

    def seed(self, seed=None):
        """Sets the random seed for reproducibility."""
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.np_random is None:  # Initialize the random number generator if not set
            self.seed(42)
        self.price = 100
        self.time_step = 0
        self.position = 0
        self.strike = self.np_random.choice(self.strike_prices)
        return self._get_state()

    def step(self, action):
        action -= 2
        prev_position = self.position
        self.position += action

        trade_cost = self.cost_multiplier * abs(action)

        self.price *= np.exp(self.np_random.normal(0, 0.01))

        option_payoff = max(self.price - self.strike, 0)
        reward = -(trade_cost + (self.position * self.price - option_payoff) ** 2)

        self.time_step += 1
        done = self.time_step >= self.total_steps
        return self._get_state(), reward, done, {}

    def _get_state(self):
        time_to_maturity = self.total_steps - self.time_step
        return np.array([self.price, time_to_maturity, self.position, self.strike], dtype=np.float32)
