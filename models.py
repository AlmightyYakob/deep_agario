from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


def base_conv_model(env):
    input_shape = env.observation_space.shape

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

    return model


def full_combined_conv_lstm_model(env):
    # print(env.observation_space.shape)
    input_shape = (1, *env.observation_space.shape)

    model = Sequential()
    model.add(
        ConvLSTM2D(
            filters=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            input_shape=input_shape,
            padding="same",
            return_sequences=True,
        )
    )
    model.add(BatchNormalization())

    model.add(
        ConvLSTM2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            return_sequences=True,
        )
    )
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(env.action_space.n))
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"],
    )
    return model


def base_enhanced_conv_model(env):
    input_shape = env.observation_space.shape

    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            activation="relu",
            input_shape=input_shape,
        )
    )
    # model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    return model


def enhanced_conv_lstm_model(env):
    model = base_enhanced_conv_model(env)
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(env.action_space.n, activation="softmax"))
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"],
    )
    return model


def conv_model(env):
    num_actions = env.action_space.n

    model = base_conv_model(env)
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_actions, activation="softmax"))
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"],
    )
    return model


def lstm_conv_model(env):
    model = Sequential()
    model.add(
        TimeDistributed(
            base_conv_model(env), input_shape=(1, *env.observation_space.shape)
        )
    )
    print(model.output.shape)
    # model.add(Flatten())
    model.add(LSTM(256))
    model.add(Dense(128))
    model.add(Dense(env.action_space.n))
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"],
    )
    return model
