import gym
import gym_io # NOQA


env = gym.make("slitherio-v0")
obs = env.reset()
res = env.step(0)
res = env.step(0)
