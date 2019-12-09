import gym
import gym_io  # NOQA

import numpy as np

from keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

# from keras import backend as K
# K.set_image_data_format('channels_first')


DQN_MEMORY_SIZE = 100


class CustomProcessor(Processor):
    """
    acts as a coupling mechanism between the agent and the environment
    """

    def process_state_batch(self, batch):
        """
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        """
        squeezed = np.squeeze(batch, axis=1)
        reshaped = np.reshape(squeezed, newshape=(*squeezed.shape, 1))
        return reshaped


def conv_model(env):
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_actions, activation="softmax"))

    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"],
    )
    return model


try:
    # env = gym.make("AirRaid-v0")
    env = gym.make("slitherio-v0")
    # env = gym.make("slitherio-v0", headless=False, width=500, height=500)

    model = conv_model(env)
    # print(model.summary())

    policy = EpsGreedyQPolicy(eps=0.1)
    memory = SequentialMemory(limit=DQN_MEMORY_SIZE, window_length=1)
    dqn = DQNAgent(
        model=model,
        nb_actions=env.action_space.n,
        memory=memory,
        nb_steps_warmup=DQN_MEMORY_SIZE,
        target_model_update=1e-2,
        policy=policy,
        processor=CustomProcessor(),
    )
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])
    dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

    env.reset()
    dqn.test(env, nb_episodes=5, visualize=True)

    env.close()

except Exception as e:
    print(e)
    env.driver.quit()
