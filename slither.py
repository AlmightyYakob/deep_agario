import gym
import gym_io # NOQA


# env = gym.make("slitherio-v0")
env = gym.make("slitherio-v0", headless=False)
obs = env.reset()

while True:
    obs, r, done, _ = env.step(env.action_space.sample())

    if done:
        break
