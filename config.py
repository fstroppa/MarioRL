# restricted_actions_dict = {'B': 1, 'Y': 2, 'A': 256, 'LEFT': 64, 'RIGHT': 128, 'UP': 16, 'DOWN': 32, 'Y+RIGHT': 130}
restricted_actions_dict = {'Y+RIGHT': 130, 'B+Y+RIGHT':131}

HM_EPISODES = 25000
SHOW_EVERY = 100
SAVE_OUTPUT = False
OUTPUT_NAME = 'actions_array_donut_plains_1_4.pkl'
SHOW_FINAL_OUTPUT = True

EPSILON = 0.9
EPS_DECAY = 0.95
NUMBER_OF_FRAMES_PER_ACTION = 4
SAFE_FRAMES = int(240 / NUMBER_OF_FRAMES_PER_ACTION)
DEAD_OR_STUCK = int(80 / NUMBER_OF_FRAMES_PER_ACTION)

