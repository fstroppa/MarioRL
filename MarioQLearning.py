import numpy as np
import retro
import tensorflow as tf

from config import *


restricted_actions_list = np.array(list(restricted_actions_dict.values()))

def create_env():
    env = retro.make(game='SuperMarioWorld-Snes', state='DonutPlains1.state')
    env.reset()
    _, _, done, info = env.step([np.random.choice(restricted_actions_list)])
    return env, done, info


env, done, info = create_env()




q_table = np.random.random(size=(400 * 120, 2))
episode_rewards = []


class Episode:
    def __init__(self, env, episode_number):
        self.env = env
        self.show = self.show_episode(episode_number)
        self.env.reset()
        _, _, self.done, self.info = env.step([restricted_actions_list[0]])
        self.frame = 0
        self.episode_reward = 0
        self.done = False
        self.last_x = []
        self.last_key = []

    def show_episode(self, episode_number):
        if episode_number % SHOW_EVERY == 0:
            print(f"on #{episode_number}, epsilon is {epsilon}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
        else:
            show = False
        return show

    def get_action(self):
        if np.random.random() > epsilon or self.show:
            action = np.argmax(q_table[self.frame])
        else:
            action = np.random.randint(0, 1)
        return action

    def run(self):
        key = self.frame
        action = self.get_action()


        _, _, self.done, self.info = env.step([restricted_actions_list[action]])
        _, _, self.done, self.info = env.step([restricted_actions_list[action]])
        _, _, self.done, self.info = env.step([restricted_actions_list[action]])
        _, _, self.done, self.info = env.step([restricted_actions_list[action]])

        if self.show:
            episode.env.render()
        rew = self.info['x'] - 0.5

        self.last_key.append((key, action))
        self.last_x.append(self.info['x'])
        if self.last_x[-1] == min(self.last_x[-10:]) and self.frame > 30:
            rew = -20000
            self.done = True

            for tuple_key_action in self.last_key[:-30]:
                q_table[tuple_key_action[0]][tuple_key_action[1]] += 10

            # i = 1
            # for tuple_key_action in self.last_key[-10:]:
            #     q_table[tuple_key_action[0]][tuple_key_action[1]] -= rew * i
            #     i *= 0.9
        self.frame += 1

        new_key = self.frame
        max_future_q = np.max(q_table[new_key])
        current_q = q_table[key][action]

        if info['win'] > 0:
            new_q = 100000
            self.done = True
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (rew + DISCOUNT * max_future_q)

        q_table[key][action] = new_q

        self.episode_reward += rew




for episode_number in range(HM_EPISODES):
    episode = Episode(env, episode_number)

    while not episode.done:
        episode.run()

    episode_rewards.append(episode.episode_reward)
    epsilon *= EPS_DECAY
