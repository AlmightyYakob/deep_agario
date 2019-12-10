import gym
import gym_io
from keras.models import load_model
from slither import conv_model, CustomProcessor, DQN_MEMORY_SIZE

from keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


env = gym.make("slitherio-v0", headless=False, width=500, height=500)
model = conv_model(env)
# model = load_model("current_model.h5")
print(model.weights)
model.load_weights("current_model.h5")
print(model.weights)
policy = EpsGreedyQPolicy(eps=0.2)
memory = SequentialMemory(limit=DQN_MEMORY_SIZE, window_length=1)
dqn = DQNAgent(
    model=model,
    nb_actions=env.action_space.n,
    memory=memory,
    # nb_steps_warmup=DQN_MEMORY_SIZE,
    target_model_update=1e-2,
    policy=policy,
    processor=CustomProcessor(),
)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])

dqn.test(env, nb_episodes=5, visualize=True)
