# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import datetime

import gym
import os

import numpy as np
from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from utils import SUMMARY_FILE_PATH, REWARDS_FILE_PATH, STEPS_FILE_PATH, PLOT_SAVE_PATH
import ware_house.envs


def test_gym(environment):
    episodes = 10
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
    model = A2C("MultiInputPolicy", environment, verbose=1, n_steps=100)
    model.learn(total_timesteps=time_step)

    # evaluate_policy(model, environment, n_eval_episodes=10, render=True)

env_versions = ["1", "2", "3", "4"]
learning_rates = [0.7]
time_steps = [5000, 50000, 500000, 5000000]
def train_all():
    for env_version in env_versions:
        for learning_rate in learning_rates:
            for time_step in time_steps:
                with open(SUMMARY_FILE_PATH, 'a+', encoding='UTF8', newline='') as f:
                    f.write(f"\nVersion number : {env_version}\n")
                    f.write(f"Learning rate : {learning_rate}\n")
                    f.write(f"Time steps : {time_step}\n")
                    f.close()
                env = gym.make(f"rware-v{env_version}")
                train(env, learning_rate=learning_rate, time_step=time_step)
                plot_data(REWARDS_FILE_PATH, STEPS_FILE_PATH, "episode",  env_version, learning_rate, time_step)


def plot_data(first_file_path, second_file_path, x_axis_label, version_num, learning_rate, time_step):
    ## open cur_rewards.txt file read rewards
    rewards = []
    plt.clf()
    if os.path.exists(first_file_path):
        with open(first_file_path, "r") as f:
            rewards_str = f.read().split("\n")
            rewards_tmp = filter(lambda line: line != '', rewards_str)
            rewards = [float(line) for line in rewards_tmp]
            print(rewards)
            ## count reward numbers
            episodes_num = np.arange(0, len(rewards), 1)
            ## plot
            plt.plot(episodes_num, rewards, '-', color='orange', label='rewards')
            plt.title(f"Agent number: {version_num} lr: {learning_rate} ts: {time_step}")
            plt.xlabel(x_axis_label)
            ## save graph
            delete_file(first_file_path)
    else:
        print("The file does not exist")

    if os.path.exists(second_file_path):
        with open(second_file_path, "r") as f:
            rewards_str = f.read().split("\n")
            rewards_tmp = filter(lambda line: line != '', rewards_str)
            rewards = [float(line) for line in rewards_tmp]
            print(rewards)
            ## count reward numbers
            episodes_num = np.arange(0, len(rewards), 1)
            ## plot
            plt.plot(episodes_num, rewards, '-', color='blue', label='steps')
            plt.title(f"Agent number: {version_num} lr: {learning_rate} ts: {time_step}")
            plt.legend()
            plt.savefig(f"{PLOT_SAVE_PATH}/{x_axis_label}_{version_num}_{learning_rate}_{time_step}.png")
            ## save graph
            delete_file(second_file_path)
    else:
        print("The file does not exist")


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file does not exist")




if __name__ == '__main__':
    delete_file(SUMMARY_FILE_PATH)
    delete_file(REWARDS_FILE_PATH)
    delete_file(STEPS_FILE_PATH)
    PLOT_SAVE_PATH = f"{PLOT_SAVE_PATH}_{datetime.datetime.now()}"
    os.mkdir(PLOT_SAVE_PATH)
    train_all()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
