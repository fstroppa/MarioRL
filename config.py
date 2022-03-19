# Configuration paramaters for the whole setup
seed = 42
# restricted_actions_dict = {'B': 1, 'Y': 2, 'A': 256, 'LEFT': 64, 'RIGHT': 128, 'UP': 16, 'DOWN': 32, 'Y+RIGHT': 130}
restricted_actions_dict = {'Y+RIGHT': 130, 'B+Y+RIGHT':131}
HM_EPISODES = 250000
SHOW_EVERY = 10

epsilon = 0.9
EPS_DECAY = 0.995
NUMBER_OF_FRAMES_PER_ACTION = 6
SAFE_FRAMES = int(180 / NUMBER_OF_FRAMES_PER_ACTION)
DEAD_OR_STUCK = int(60 / NUMBER_OF_FRAMES_PER_ACTION)


LEARNING_RATE = 0.5
DISCOUNT = 0.90