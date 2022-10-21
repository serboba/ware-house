# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import gym
import os
from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import ware_house.envs

LOG_PATH = ""


def test_gym(environment):
    episodes = 1
    for episode in range(1, episodes + 1):
        state, info = environment.reset()
        done = False
        score = 0
        # print(info)
        while not done:
            action = environment.action_space.sample()
            n_state, reward, done, info = environment.step(action)
            score += reward
            # print(f"{info}")
        print('Episode:{} Score:{}'.format(episode, score))
    environment.close()


def train(environment, learning_rate, time_step):
    model = A2C("MultiInputPolicy", environment, learning_rate=learning_rate, verbose=1)
    model.learn(total_timesteps=time_step)
    # evaluate_policy(model, environment, n_eval_episodes=10, render=True)


env_versions = ["1", "2", "3", "4"]
learning_rates = [0.05, 0.1, 0.5, 0.7]
time_steps = [50, 100, 1000, 5000]

if __name__ == '__main__':

    for env_version in env_versions:
        for learning_rate in learning_rates:
            for time_step in time_steps:
                with open('warehouse.txt', 'a', encoding='UTF8', newline='') as f:
                    f.write(f"Version number : {env_version}")
                    f.write(f"Learning rate : {learning_rate}")
                    f.write(f"Time steps : {time_step}")
                    f.close()
                env = gym.make(f"rware-v{env_version}")
                train(env, learning_rate=learning_rate, time_step=time_step)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
