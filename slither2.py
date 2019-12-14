# -*- coding: utf-8 -*-
from ddqn import DQNAgent

import numpy as np
import gym
import gym_io
import tensorflow as tf

EPISODES = 5000

if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    env = gym.make("slitherio-v0", headless=False, width=500, height=585)

    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_shape, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            print(state.shape)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
