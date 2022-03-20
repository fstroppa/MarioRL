import pickle
import time
import numpy as np
import retro
from config import *
from episode import Episode


def create_env():
    env = retro.make(game='SuperMarioWorld-Snes', state='DonutPlains1.state')
    env.reset()
    _, _, done, info = env.step([np.random.choice(RESTRICTED_ACTIONS)])
    return env, done, info


def render_final_output(episode):
    episode.env.reset()
    episode.env.step([RESTRICTED_ACTIONS[0]])
    for action in episode.actions_taken:
        episode.take_actions(action)
        episode.env.render()
        time.sleep(0.02)
        if done:
            time.sleep(2)
            break


def get_winnable_episode():
    safe_frame = 0
    epsilon = EPSILON
    for episode_number in range(HM_EPISODES):
        episode = Episode(env, episode_number, safe_frame, actions_array)

        while not episode.done:
            episode.run_one_step(epsilon)

        safe_frame = episode.safe_frame
        print(f'Run number {episode_number}: '
              f'{safe_frame * NUMBER_OF_FRAMES_PER_ACTION}'
              f' safe frames advanced.')

        epsilon *= EPS_DECAY
        if episode.win:
            return episode


if __name__ == '__main__':
    env, done, info = create_env()
    actions_array = np.random.randint(NUMBER_OF_POSSIBLE_ACTIONS,
                                      size=NUMBER_OF_POSSIBLE_FRAMES)

    episode = get_winnable_episode()

    if SHOW_FINAL_OUTPUT:
        render_final_output(episode)

    if SAVE_OUTPUT:
        with open(OUTPUT_NAME, 'wb') as output:
            pickle.dump(episode.actions_array, output, 1)