from gym import Env, spaces


class SlitherIOEnv(Env):
    metadata = {"render.modes": ["human"]}

    # TBD, need to figure out full observation space
    # observation_space =
    # observation_space.shape =

    # Representing degrees to move mouse, might change
    action_space = spaces.Discrete(360)
    action_space.shape = (1, 360)

    def __init__(self):
        # Initialize selenium, open website but don't click play yet
        # Apply required CSS to dom elements to hide overlays and stuff
        pass

    def step(self, action):
        # action is a
        # screenshot with selenium              (state)
        # Get current score = r0 with selenium
        # Take provided action with selenium    (action)
        # Get current score = r1 with selenium
        # Calculate reward = r1 - r0            (reward)
        # screenshot with selenium              (next state)
        pass

    def reset(self):
        # Basically just call __init__
        pass

    def render(self, mode="human", close=False):
        # Probably ignore for now
        pass
