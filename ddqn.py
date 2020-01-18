# -*- coding: utf-8 -*-
import random
import time

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
import numpy as np
import cv2


def reshape_state(s):
    return s.reshape(1, s.shape[0], s.shape[0], 1)


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        #model = Sequential()
        #model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(24, activation='relu'))
        #model.add(Dense(self.action_size, activation='linear'))
        #model.compile(loss=self._huber_loss,
        #              optimizer=Adam(lr=self.learning_rate))
        model = Sequential()

        state_shape = (self.state_shape[0], self.state_shape[1], 1)

        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=state_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))

        model.add(Flatten())

        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_size, activation="softmax"))

        model.compile(
            loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"],
        )

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = reshape_state(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def test(self, state):
        state = reshape_state(state)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        # Prepare states
        states = []
        next_states = []
        for state, action, reward, next_state, done in minibatch:
            # Do nothing
            states.append(state.reshape(self.state_shape))

            # Flip horizontally
            horz = cv2.flip(state, 1)
            states.append(horz.reshape(self.state_shape))

            # Flip vertically
            vert = cv2.flip(state, 0)
            states.append(vert.reshape(self.state_shape))

            # Diagonal
            diag = cv2.flip(horz, 0)
            states.append(diag.reshape(self.state_shape))

            # Add next states is appliciable
            if not done:
                # Do nothing
                next_states.append(next_state.reshape(self.state_shape))

                # Flip horizontally
                horz = cv2.flip(next_state, 1)
                next_states.append(horz.reshape(self.state_shape))

                # Flip vertically
                vert = cv2.flip(next_state, 0)
                next_states.append(vert.reshape(self.state_shape))

                # Diagonal
                diag = cv2.flip(horz, 0)
                next_states.append(diag.reshape(self.state_shape))

        states = np.array(states)
        next_states = np.array(next_states)

        # Get predictions for each state
        targets = self.model.predict(states)

        # Get predictions for next states
        next_rewards = self.model.predict(next_states)

        # Go through and set each target[action] to the true reward value
        j = 0

        for i in range(batch_size):
            reward = minibatch[i][2]

            # If not a final state
            if not minibatch[i][4]:
                # Apply gamma to next state
                reward = minibatch[i][2] + self.gamma * np.amax(next_rewards[j])
                j += 1

            # Do nothing
            action = minibatch[i][1]
            targets[i][action] = reward

            # Horizontal
            horz_action = (4 - action) % 8
            targets[i][horz_action] = reward

            # Verical
            vert_action = (8 - action) % 8
            targets[i][vert_action] = reward

            # Diagonal
            diag_action = (4 + action) % 8
            targets[i][diag_action] = reward

        # Train model to fit new target
        self.model.fit(states, targets, epochs=1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        '''
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        '''

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

