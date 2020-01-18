# -*- coding: utf-8 -*-
from ddqn import DQNAgent

import gym
import gym_io
import numpy as np
import time

EPISODES = 100

if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    env = gym.make("slitherio-v0", headless=False, width=500, height=500)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_shape, action_size)
    #agent.load("./save/snek.h5")
    done = False
    batch_size = 256

    for e in range(EPISODES):
        state = env.reset()
        ticks = 0

        while not done:
            #if ticks % 50 == 0:
            #    start_time = time.time()

            #env.render()
            action = agent.test(state)
            next_state, reward, done, _ = env.step(action)
            #agent.remember(state, action, reward, next_state, done)
            state = next_state

            #if ticks % 50 == 0:
            #    print("--- %.2f --- " % (1.0 / (time.time() - start_time)))

            ticks += 1

            #if done:
            #    # agent.update_target_model()
            #    agent.replay(min(256, len(agent.memory)))
            #    print("episode: {}/{}, score: {}, e: {:.2}"
            #          .format(e, EPISODES, ticks, agent.epsilon))
            #    done = False
            #    break
            # if len(agent.memory) > batch_size:
            #    agent.replay(batch_size)
        #if e % 10 == 0:
        #    agent.save("./save/snek.h5")
