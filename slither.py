import gym
import gym_io  # NOQA

import numpy as np

from keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint

from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

# from keras import backend as K
# K.set_image_data_format('channels_first')


DQN_MEMORY_SIZE = 100
MODEL_SAVE_STEP_INTERVAL = 50
# SAVED_MODEL_NAME = "current_model.h5"
SAVED_MODEL_NAME = "new_current_model.h5"
NSTEPS = 10000


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


class LSTMProcessor(Processor):
    """
    acts as a coupling mechanism between the agent and the environment
    """

    def process_state_batch(self, batch):
        """
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        """
        return np.reshape(batch, newshape=(*batch.shape, 1))


def base_conv_model(env):
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
    return model


def conv_model(env):
    model = base_conv_model(env)
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"],
    )
    return model


def lstm_conv_model(env):
    model = Sequential()
    model.add(TimeDistributed(base_conv_model(env), input_shape=(1, *env.observation_space.shape)))
    model.add(LSTM(256))
    model.add(Dense(128))
    model.add(Dense(env.action_space.n))
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"],
    )
    return model


def main():
    try:
        # env = gym.make("AirRaid-v0")
        # env = gym.make("slitherio-v0")
        env = gym.make("slitherio-v0", headless=False, width=500, height=500)

        model_callbacks = [
            ModelIntervalCheckpoint(
                SAVED_MODEL_NAME, interval=MODEL_SAVE_STEP_INTERVAL, verbose=0
            )
        ]

        # model = conv_model(env)
        model = lstm_conv_model(env)
        # print(model.summary())

        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(eps=0.1),
            attr="eps",
            value_max=1.0,
            value_min=0.1,
            value_test=0.1,
            nb_steps=NSTEPS,
        )
        memory = SequentialMemory(limit=DQN_MEMORY_SIZE, window_length=1)
        dqn = DQNAgent(
            model=model,
            nb_actions=env.action_space.n,
            memory=memory,
            # nb_steps_warmup=DQN_MEMORY_SIZE,
            target_model_update=1e-2,
            policy=policy,
            enable_double_dqn=True,
            # enable_dueling_network=True,
            # dueling_type="avg",
            # processor=CustomProcessor(),
            processor=LSTMProcessor(),
        )
        dqn.compile(Adam(lr=1e-3), metrics=["mae"])
        dqn.fit(
            env, nb_steps=NSTEPS, visualize=False, verbose=1, callbacks=model_callbacks
        )

        env.reset()
        dqn.test(env, nb_episodes=5, visualize=True)

        env.close()

    except Exception as e:
        env.close()
        print(e)


if __name__ == "__main__":
    main()
