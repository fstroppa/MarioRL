import numpy as np
import retro
import tensorflow as tf

from config import *

## Create ENV

env = retro.make(game='SuperMarioWorld-Snes', state='DonutPlains1.state')
flatten = env.reset()
restricted_actions_list = list(restricted_actions_dict.values())
_, _, done, info = env.step([np.random.choice(restricted_actions_list)])

## Create ENV







q_table = {}
episode_rewards = []
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
            # time.sleep(1/250)
            key = frame
            if key not in q_table:
                q_table[key] = [np.random.uniform(0, 1) for i in range(2)]
            action = np.argmax(q_table[key])
            ob, rew, done, info = env.step([restricted_actions_list[action]])
            last_x.append(info['x'])
            frame += 1
            if last_x[-1] == min(last_x[-10:]) and frame > 10:
                break

        else:
            key = frame
            if key not in q_table:
                q_table[key] = [np.random.uniform(0, 1) for i in range(2)]
                action = np.argmax(q_table[key])

            else:

                if np.random.random() > epsilon:
                    action = np.argmax(q_table[key])
                else:
                    action = np.random.randint(0, 1)

            last_key.append((key, action))

            ob, rew, done, info = env.step([restricted_actions_list[action]])
            last_x.append(info['x'])
            if last_x[-1] == min(last_x[-10:]) and frame > 10:
                rew -= 100
            frame += 1

            new_key = frame
            if new_key not in q_table:
                q_table[new_key] = [np.random.uniform(0, 1) for i in range(2)]
            max_future_q = np.max(q_table[new_key])
            current_q = q_table[key][action]

            if info['win'] > 0:
                new_q = 100000
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (rew + DISCOUNT * max_future_q)
            q_table[key][action] = new_q

            episode_reward += rew
            if info['win'] > 0:
                break

            if rew < 0:
                for tuple_key_action in last_key[-30:]:
                    q_table[tuple_key_action[0]][tuple_key_action[1]] = q_table[tuple_key_action[0]][tuple_key_action[1]] - 10
                break





    # print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY



#
#
# for i in range(1000):
#     ob, rew, done, info = env.step([np.random.choice(restricted_actions_list)])
#
#     result = cv2.matchTemplate(ob, mario_img, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#
#     f = env.render(mode="rgb_array")
#     # Image.fromarray(f).show()
#
#     w = mario_img.shape[1]
#     h = mario_img.shape[0]
#     # cv2.rectangle(ob, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,255), 2)
#     cv2.rectangle(ob, (info['x'], info['y']), (info['x'] + 22, info['y'] + 15), (0,255,255), 2)
#     print(info)
#
#
#     cv2.imshow('Mario', env.render(mode="human"))
#     wait = cv2.waitKey(5)
#
