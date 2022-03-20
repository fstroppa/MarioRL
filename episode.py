import numpy as np
from config import *


class Episode:
    def __init__(self, env, episode_number, safe_frame, actions_array):
        self.env = env
        self._start_environment()
        self.actions_array = actions_array
        self.safe_frame = safe_frame
        self.episode_number = episode_number

    def _start_environment(self):
        self.env.reset()
        _, _, self.done, self.info = self.env.step([RESTRICTED_ACTIONS[0]])
        self.frame = 0
        self.done = False
        self.last_x_positions = []
        self.actions_taken = []
        self.win = False

    def _show_episode(self):
        return self.episode_number % SHOW_EVERY == 0

    def _frame_decision_is_safe(self):
        return self.frame < self.safe_frame

    def _get_action(self, epsilon):
        if self._frame_decision_is_safe():
            return self.actions_array[self.frame]
        else:
            if np.random.random() > epsilon:
                return self.actions_array[self.frame]
            return np.random.randint(NUMBER_OF_POSSIBLE_ACTIONS)

    def run_one_step(self, epsilon):
        action = self._get_action(epsilon)
        self.take_actions(action)
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

    def take_actions(self, action):
        for _ in range(NUMBER_OF_FRAMES_PER_ACTION):
            _, _, self.done, self.info = self.env.step(
                [RESTRICTED_ACTIONS[action]])


    def _save_losing_results(self):
        for idx, action_take in enumerate(self.actions_taken):
            if idx > self.safe_frame - SAFE_FRAMES:
                self.actions_array[idx] = action_take
        if len(self.actions_taken) - SAFE_FRAMES > self.safe_frame:
            self.safe_frame = len(self.actions_taken) - SAFE_FRAMES

    def _save_winning_results(self):
        for idx, action_take in enumerate(self.actions_taken):
            self.actions_array[idx] = action_take

    def _dead_or_stuck(self):
        return self.last_x_positions[-1] == min(self.last_x_positions[-DEAD_OR_STUCK:]) \
               and self.frame > 30