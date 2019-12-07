import gym
import gym_io  # NOQA


from keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from keras import backend as K


K.set_image_data_format('channels_first')


def conv_model(env):
    input_shape = (1,) + env.observation_space.shape[:2]
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
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=["accuracy"],
    )
    return model

try:
    env = gym.make("slitherio-v0")
    # env = gym.make("slitherio-v0", headless=False, width=600, height=600)

    model = conv_model(env)
    # print(model.summary())

    policy = EpsGreedyQPolicy(eps=0.1)
    memory = SequentialMemory(limit=100, window_length=1)
    dqn = DQNAgent(
        model=model,
        nb_actions=env.action_space.n,
        memory=memory,
        nb_steps_warmup=100,
        target_model_update=1e-2,
        policy=policy,
    )
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])
    dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

    env.reset()
    dqn.test(env, nb_episodes=5, visualize=True)

    env.close()

except Exception as e:
    print(e)
    env.driver.quit()
