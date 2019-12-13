import gym
import gym_io  # NOQA

import click
import numpy as np

from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint

from selenium.common.exceptions import WebDriverException

from models import (
    full_combined_conv_lstm_model,
    conv_model,
    lstm_conv_model,
    enhanced_conv_lstm_model,
)


# from keras import backend as K
# K.set_image_data_format('channels_first')


DQN_MEMORY_SIZE = 100
MODEL_SAVE_STEP_INTERVAL = 100
# SAVED_MODEL_NAME = "current_model.h5"
SAVED_MODEL_NAME = "new_current_model.h5"
NSTEPS = 100000


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


# @click.option("-l", "--load-model", "model_name")
# def train(model_name):
#     if model_name:
#         model =
#     else:
#         model =


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
        # model = lstm_conv_model(env)
        model = full_combined_conv_lstm_model(env)
        model.load_weights(SAVED_MODEL_NAME)
        # model = enhanced_conv_lstm_model(env)
        # print(model.summary())
        major_rounds = int(NSTEPS / 1000)
        max_total_eps = 1.0
        min_total_eps = 0.1
        eps_range = max_total_eps - min_total_eps
        eps_step = eps_range / major_rounds

        for major_step in range(major_rounds):
            print("Major step", major_step, "of", major_rounds)

            max_eps = max_total_eps - eps_step * major_step
            min_eps = max_eps - eps_step

            policy = LinearAnnealedPolicy(
                EpsGreedyQPolicy(eps=0.1),
                attr="eps",
                value_max=max_eps,
                value_min=min_eps,
                value_test=0.1,
                nb_steps=1000,
            )
            memory = SequentialMemory(limit=DQN_MEMORY_SIZE, window_length=1)
            dqn = DQNAgent(
                model=model,
                nb_actions=env.action_space.n,
                memory=memory,
                target_model_update=1e-2,
                policy=policy,
                enable_double_dqn=True,
                processor=LSTMProcessor(),
            )
            dqn.compile(Adam(lr=1e-3), metrics=["mae"])
            dqn.fit(
                env,
                nb_steps=1000,
                visualize=False,
                verbose=1,
                callbacks=model_callbacks,
                log_interval=1000,  # TODO bruh this fixes the 10000 issue!
            )

        env.reset()
        dqn.test(env, nb_episodes=5, visualize=True)

        env.close()

    except WebDriverException as e:
        print(e)

    except Exception as e:
        env.close()
        print(e)


if __name__ == "__main__":
    main()
