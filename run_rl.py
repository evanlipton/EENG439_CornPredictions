import pandas as pd
import numpy as np
import importlib

import pickle

from gym_env import FuturesEnv

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np


def train_and_test(data_pkl="data_lagged.pkl", price_pkl="close_prices.pkl"):
    NUM_BUSHELS = 50000
    NUM_TIMESTEPS = 60000

    with open(data_pkl, "rb") as f:
        dataset = pickle.load(f)

    with open(price_pkl, "rb") as f:
        prices = pickle.load(f)

    for test_year in dataset.keys():
	do_single_year(dataset, prices, test_year)
        train_env = [lambda: FuturesEnv(
            [dataset[y] for y in dataset.keys() if y != test_year],
            NUM_BUSHELS,
            [prices[y] for y in dataset.keys() if y != test_year])]

        env = DummyVecEnv(train_env)

        model = DQN(MlpPolicy, env)
        model.learn(total_timesteps=NUM_TIMESTEPS)

        test_env = [lambda: FuturesEnv(
            [dataset[test_year]], NUM_BUSHELS, [prices[test_year]])]
        test = DummyVecEnv(test_env)

        obs = test.reset()
        done = False
        sell_info = []
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = test.step(action)
            info = info[0]
            if action > 0:
                sell_info += [(str(info['date']),
                               str(info['price']), str(info['amount']))]
            # test.render()

        print("Finished ", str(test_year))

        with open("DQN-" + str(test_year) + ".csv", "w") as f:
            f.write("Sell Date, Sell Price, Sell Amount\n")
            [f.write(x[0] + "," + x[1] + "," + x[2] + "\n") for x in sell_info]

def do_single_year(dataset, prices, test_year):
    NUM_BUSHELS = 50000
    NUM_TIMESTEPS = 300000

    train_env = [lambda: FuturesEnv(
	[dataset[y] for y in dataset.keys() if y != test_year],
	NUM_BUSHELS,
	[prices[y] for y in dataset.keys() if y != test_year])]

    env = DummyVecEnv(train_env)

    model = DQN(MlpPolicy, env)
    model.learn(total_timesteps=NUM_TIMESTEPS)

    test_env = [lambda: FuturesEnv(
	[dataset[test_year]], NUM_BUSHELS, [prices[test_year]])]
    test = DummyVecEnv(test_env)

    obs = test.reset()
    done = False
    sell_info = []
    while not done:
	action, _states = model.predict(obs)
	obs, rewards, done, info = test.step(action)
	info = info[0]
	if action > 0:
	    sell_info += [(str(info['date']),
			   str(info['price']), str(info['amount']))]
	# test.render()

    print("Finished ", str(test_year))

    with open("DQN-" + str(test_year) + ".csv", "w") as f:
	f.write("Sell Date, Sell Price, Sell Amount\n")
	[f.write(x[0] + "," + x[1] + "," + x[2] + "\n") for x in sell_info]


