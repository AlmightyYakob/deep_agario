import gym
import gym_io # NOQA


# env = gym.make("slitherio-v0")
env = gym.make("slitherio-v0", headless=False)
obs = env.reset()

while True:
    res = env.step(env.action_space.sample())
