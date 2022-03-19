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


def get_action():
    if np.random.random() > epsilon:
        action = np.argmax(q_table[key])
    else:
        action = np.random.randint(0, 1)
    return action


def show_episode():
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    return show


for episode in range(HM_EPISODES):
    show = show_episode()

    env.reset()
    ob, rew, done, info = env.step([restricted_actions_list[0]])
    frame = 0
    episode_reward = 0
    done = False
    win = False
    last_x = []
    last_key = []

    while not done or not win:
        if show:
            env.render()
            key = frame
            action = np.argmax(q_table[key])
            _, _, done, info = env.step([restricted_actions_list[action]])
            last_x.append(info['x'])
            frame += 1

            if last_x[-1] == min(last_x[-10:]) and frame > 10:
                break

        else:
            key = frame

            action = get_action()

            last_position = info['x']
            _, _, done, info = env.step([restricted_actions_list[action]])
            rew = info['x'] - last_position - 0.5

            last_key.append((key, action))
            last_x.append(info['x'])
            if last_x[-1] == min(last_x[-30:]) and frame > 30:
                rew = -100
                done = True
            frame += 1

            new_key = frame
            max_future_q = np.max(q_table[new_key])
            current_q = q_table[key][action]

            if info['win'] > 0:
                new_q = 100000
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (rew + DISCOUNT * max_future_q)
            q_table[key][action] = new_q

            episode_reward += rew
            if info['win'] > 0:
                win = True

            if done:
                i = 1
                for tuple_key_action in last_key[-30:]:
                    q_table[tuple_key_action[0]][tuple_key_action[1]] = q_table[tuple_key_action[0]][tuple_key_action[1]] - 10 / i
                    i += 1
                break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
