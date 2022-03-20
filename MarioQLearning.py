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
        self._start_environment()
        self.safe_frame = safe_frame
        self.episode_number = episode_number

    def _start_environment(self):
        self.env.reset()
        _, _, self.done, self.info = self.env.step([restricted_actions_list[0]])
        self.frame = 0
        self.done = False
        self.last_x_positions = []
        self.actions_taken = []
        self.win = False

    def _show_episode(self):
        return self.episode_number % SHOW_EVERY == 0

    def _frame_decision_is_safe(self):
        return self.frame < self.safe_frame

    def _get_action(self):
        if self._frame_decision_is_safe():
            return actions_array[self.frame]
        else:
            if np.random.random() > epsilon:
                return actions_array[self.frame]
            return np.random.randint(NUMBER_OF_POSSIBLE_ACTIONS)

    def run_one_step(self):
        action = self._get_action()
        self._take_actions(action)
        self.actions_taken.append(action)
        self.frame += 1
        self.last_x_positions.append(self.info['x'])

        if self._show_episode():
            self.env.render()

        if self.info['win'] > 0:
            self.done = True
            self.win = True
            self._save_winning_results()

        if self._dead_or_stuck():
            self.done = True
            self._save_losing_results()

    def _take_actions(self, action):
        for _ in range(NUMBER_OF_FRAMES_PER_ACTION):
            _, _, self.done, self.info = self.env.step(
                [restricted_actions_list[action]])


    def _save_losing_results(self):
        for idx, action_take in enumerate(self.actions_taken):
            if idx > self.safe_frame - SAFE_FRAMES:
                actions_array[idx] = action_take
        if len(self.actions_taken) - SAFE_FRAMES > self.safe_frame:
            self.safe_frame = len(self.actions_taken) - SAFE_FRAMES

    def _save_winning_results(self):
        for idx, action_take in enumerate(self.actions_taken):
            actions_array[idx] = action_take

    def _dead_or_stuck(self):
        return self.last_x_positions[-1] == min(self.last_x_positions[-DEAD_OR_STUCK:]) \
               and self.frame > 30


if __name__ == '__main__':
    restricted_actions_list = np.array(list(restricted_actions_dict.values()))
    env, done, info = create_env()
    actions_array = np.random.randint(NUMBER_OF_POSSIBLE_ACTIONS,
                                      size=NUMBER_OF_POSSIBLE_FRAMES)

    safe_frame = 0
    epsilon = EPSILON
    for episode_number in range(HM_EPISODES):

        episode = Episode(env, episode_number, safe_frame)

        while not episode.done:
            episode.run_one_step()
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