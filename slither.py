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


def conv_model(env, frame_stack=1):
    # input_shape = (frame_stack, *env.observation_space.shape)
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
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=["accuracy"],
    )
    return model


# env = gym.make("slitherio-v0")
env = gym.make("slitherio-v0", headless=True)
obs = env.reset()

model = conv_model(env)
print(model.summary())

policy = EpsGreedyQPolicy(eps=0.1)
memory = SequentialMemory(limit=10000, window_length=1)
dqn = DQNAgent(
    model=model,
    nb_actions=env.action_space.n,
    memory=memory,
    nb_steps_warmup=10,
    target_model_update=1e-2,
    policy=policy,
)

dqn.compile(Adam(lr=1e-3), metrics=["mae"])
dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

env.reset()
dqn.test(env, nb_episodes=5, visualize=True)

env.close()


while True:
    obs, r, done, _ = env.step(env.action_space.sample())

    if done:
        break
