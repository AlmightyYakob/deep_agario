import gym
import gym_io # NOQA


env = gym.make("slitherio-v0")
obs = env.reset()

while True:
    res = env.step(0)
