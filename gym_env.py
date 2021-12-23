import gym
from gym import spaces

import pandas as pd
import numpy as np
import random


class FuturesEnv(gym.Env):
    def __init__(self, datasets, bushels, prices):
        super(FuturesEnv, self).__init__()
        N_ACTIONS = 16
        self.PENALTY_PRICE = 50
        self.DAILY_REWARD = 750
        self.NO_SELL_BONUS = 1000
        self.LATE_SELL_BONUS = 3000

        self.datasets = datasets
        self.prices_datasets = prices

        x = random.randint(0, len(self.datasets)-1)
        data = self.datasets[x]
        self.prices = prices[x]

        self.data = data.drop(['Date'], axis=1)
        self.dates = data['Date']

        self.starting_bushels = bushels
        self.current_bushels = bushels
        self.balance = 0
        self.current_step = 0

        self.action_space = spaces.Discrete(N_ACTIONS)
        low = np.zeros((1, data.shape[1]))
        high = np.zeros((1, data.shape[1]))
        high = high + 1
        high[0][-1] = self.starting_bushels
        self.observation_space = spaces.Box(low=low, high=high)

    def reset(self):
        x = random.randint(0, len(self.datasets)-1)
        data = self.datasets[x]
        self.prices = self.prices_datasets[x]

        self.data = data.drop(['Date'], axis=1)
        self.dates = data['Date']
        self.current_bushels = self.starting_bushels
        self.balance = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        frame = self.data.iloc[self.current_step].to_numpy()
        obs = np.append(frame, self.current_bushels)
        return obs

    def step(self, action):
        reward, sell_amount = self._take_action(action)
        self.current_step += 1

        done = self.current_step >= len(self.data) or self.current_bushels <= 0
        obs = self._next_observation()
        if done:
            reward -= self.current_bushels * self.PENALTY_PRICE

        return obs, reward, done, {
            'date': self.dates.iloc[self.current_step-1],
            'price': self.prices['Close'][self.current_step-1],
            'amount': sell_amount
        }

    def _take_action(self, action):
        sell_amount = action * 1000
        if sell_amount > self.current_bushels:
            sell_amount = self.current_bushels

        self.current_bushels -= sell_amount
        reward = sell_amount * self.prices['Close'][self.current_step]

        self.balance += reward

        if sell_amount == 0:
            reward += self.DAILY_REWARD
        if self.current_bushels == self.starting_bushels:
            reward += self.NO_SELL_BONUS
        if self.current_step > len(self.data)/3 and sell_amount > 0:
            reward += self.LATE_SELL_BONUS
        return reward, sell_amount

    def render(self, mode='human', close=False):
        bushels_owned = self.current_bushels
        bushels_sold = self.starting_bushels - bushels_owned
        profit = self.balance

        print("Bushels sold: ", bushels_sold)
        print("Bushels remaining: ", bushels_owned)
        print("Money made: ", profit)
        print("Day %d/%d\n" % (self.current_step, len(self.data)))
