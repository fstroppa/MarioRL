import numpy as np
import retro
import tensorflow as tf

from config import *

## Create ENV
restricted_actions_list = np.array(list(restricted_actions_dict.values()))

def create_env():
    env = retro.make(game='SuperMarioWorld-Snes', state='DonutPlains1.state')
    env.reset()
    _, _, done, info = env.step([np.random.choice(restricted_actions_list)])
    return env, done, info


env, done, info = create_env()

## Create ENV


# q_table = {}
q_numpy = np.random.random(size=(400*120, 2))
episode_rewards = []


def get_action():
    if np.random.random() > epsilon:
        action = np.argmax(q_numpy[key])
    else:
        action = np.random.randint(0, 1)
    return action


for episode in range(HM_EPISODES):
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    env.reset()
    ob, rew, done, info = env.step([restricted_actions_list[0]])
    frame = 0
    episode_reward = 0
    done = False
    last_x = []
    last_key = []

    while not done:
        if show:
            env.render()
            key = frame
            action = np.argmax(q_numpy[key])
            ob, rew, done, info = env.step([restricted_actions_list[action]])
            last_x.append(info['x'])
            frame += 1

            if last_x[-1] == min(last_x[-10:]) and frame > 10:
                break

        else:
            key = frame

            action = get_action()

            last_key.append((key, action))

            ob, rew, done, info = env.step([restricted_actions_list[action]])
            last_x.append(info['x'])
            if last_x[-1] == min(last_x[-10:]) and frame > 10:
                rew -= 100
            frame += 1

            new_key = frame
            max_future_q = np.max(q_numpy[new_key])
            current_q = q_numpy[key][action]

            if info['win'] > 0:
                new_q = 100000
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (rew + DISCOUNT * max_future_q)
            q_numpy[key][action] = new_q

            episode_reward += rew
            if info['win'] > 0:
                break

            if rew < 0:
                for tuple_key_action in last_key[-30:]:
                    q_numpy[tuple_key_action[0]][tuple_key_action[1]] = q_numpy[tuple_key_action[0]][tuple_key_action[1]] - 10
                break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
