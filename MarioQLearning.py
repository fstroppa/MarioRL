import random
import time
from copy import copy

import numpy as np
import retro
import tensorflow as tf
from baselines.common.retro_wrappers import wrap_deepmind_retro

from config import *

random.seed(50)

restricted_actions_list = np.array(list(restricted_actions_dict.values()))

def create_env():
    env = retro.make(game='SuperMarioWorld-Snes', state='DonutPlains1.state')
    env.reset()
    _, _, done, info = env.step([np.random.choice(restricted_actions_list)])
    return env, done, info


env, done, info = create_env()


q_table = np.random.randint(2, size=4000)
episode_rewards = []


class Episode:
    def __init__(self, env, episode_number, safe_frame):
        self.env = env
        self.safe_frame = safe_frame
        self.show = self.show_episode(episode_number)
        self.env.reset()
        _, _, self.done, self.info = env.step([restricted_actions_list[0]])
        self.frame = 0
        self.episode_reward = 0
        self.done = False
        self.last_x = []
        self.actions_taken = []

    def show_episode(self, episode_number):
        return episode_number % SHOW_EVERY == 0

    def get_action(self):
        if self.frame < self.safe_frame:
            return q_table[self.frame]
        if np.random.random() > epsilon or self.show:
            action = q_table[self.frame]
        else:
            action = np.random.randint(2)
        return action

    def run(self):

        action = self.get_action()

        _, _, self.done, self.info = env.step([restricted_actions_list[action]])
        _, _, self.done, self.info = env.step([restricted_actions_list[action]])
        _, _, self.done, self.info = env.step([restricted_actions_list[action]])
        _, _, self.done, self.info = env.step([restricted_actions_list[action]])

        if self.show:
            episode.env.render()


        self.actions_taken.append(action)
        self.frame += 1
        self.last_x.append(self.info['x'])
        if self.last_x[-1] == min(self.last_x[-10:]) and self.frame > 30:
            self.done = True

            for idx, action_take in enumerate(self.actions_taken):
                if idx > self.safe_frame - SAFE_FRAMES:
                    q_table[idx] = action_take
            self.safe_frame = len(self.actions_taken) - SAFE_FRAMES


# env2 = copy(env)
# assert False
safe_frame = 0
for episode_number in range(HM_EPISODES):

    episode = Episode(env, episode_number, safe_frame)

    while not episode.done:
        episode.run()
    safe_frame = episode.safe_frame
    print(safe_frame)
    if episode_number % 100 == 0:
        q_table[safe_frame:] = np.random.random()
        epsilon = 0.9

    epsilon *= EPS_DECAY

