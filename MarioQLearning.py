import pickle
import time
import numpy as np
import retro
import tensorflow as tf
from config import *


def create_env():
    env = retro.make(game='SuperMarioWorld-Snes', state='DonutPlains1.state')
    env.reset()
    _, _, done, info = env.step([np.random.choice(restricted_actions_list)])
    return env, done, info





class Episode:
    def __init__(self, env, episode_number, safe_frame):
        self.env = env
        self.safe_frame = safe_frame
        self.episode_number = episode_number
        self.env.reset()
        _, _, self.done, self.info = env.step([restricted_actions_list[0]])
        self.frame = 0
        self.episode_reward = 0
        self.done = False
        self.last_x = []
        self.actions_taken = []
        self.win = False

    def show_episode(self):
        return self.episode_number % SHOW_EVERY == 0

    def get_action(self):
        if self.frame < self.safe_frame:
            return actions_array[self.frame]
        if np.random.random() > epsilon:
            action = actions_array[self.frame]
        else:
            action = np.random.randint(2)
        return action

    def run(self):
        action = self.get_action()

        for i in range(NUMBER_OF_FRAMES_PER_ACTION):
            _, _, self.done, self.info = env.step([restricted_actions_list[action]])

        if self.info['win'] > 0:
            self.done = True
            self.win = True

        if self.show_episode():
            episode.env.render()

        self.actions_taken.append(action)
        self.frame += 1
        self.last_x.append(self.info['x'])

        if self.dead_or_stuck():
            self._save_results_of_the_run()

    def _save_results_of_the_run(self):
        self.done = True
        if self.win:
            for idx, action_take in enumerate(self.actions_taken):
                actions_array[idx] = action_take
        else:
            for idx, action_take in enumerate(self.actions_taken):
                if idx > self.safe_frame - SAFE_FRAMES:
                    actions_array[idx] = action_take
            if len(self.actions_taken) - SAFE_FRAMES > self.safe_frame:
                self.safe_frame = len(self.actions_taken) - SAFE_FRAMES

    def dead_or_stuck(self):
        return self.last_x[-1] == min(self.last_x[-DEAD_OR_STUCK:]) and self.frame > 30


if __name__ == '__main__':
    restricted_actions_list = np.array(list(restricted_actions_dict.values()))
    env, done, info = create_env()
    actions_array = np.random.randint(2, size=4000)

    safe_frame = 0
    epsilon = EPSILON
    for episode_number in range(HM_EPISODES):

        episode = Episode(env, episode_number, safe_frame)

        while not episode.done:
            episode.run()
        safe_frame = episode.safe_frame
        print(f'Run number {episode_number}: {safe_frame * NUMBER_OF_FRAMES_PER_ACTION} safe frames advanced.')
        if episode_number % 20 == 0:
            actions_array[safe_frame:] = np.random.random()
            epsilon = 0.9

        epsilon *= EPS_DECAY
        if episode.win:
            actions_array = episode.actions_taken
            break

    if SHOW_FINAL_OUTPUT:
        env.reset()
        _, _, done, info = env.step([restricted_actions_list[0]])
        for action in actions_array:
            for i in range(NUMBER_OF_FRAMES_PER_ACTION):
                _, _, done, info = env.step([restricted_actions_list[action]])
            env.render()
            time.sleep(0.015)
            if done:
                time.sleep(2)
                break

    if SAVE_OUTPUT:
        with open(OUTPUT_NAME, 'wb') as output:
            pickle.dump(actions_array, output, 1)