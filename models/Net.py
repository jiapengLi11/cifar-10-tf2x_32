from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Sequential


def net_self():
    """Build a compact CNN for CIFAR-10 classification."""
    model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(32, 32, 3)),
        LeakyReLU(alpha=0.1),
        MaxPool2D(pool_size=(2, 2), strides=(1, 1)),

        Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        LeakyReLU(alpha=0.1),
        MaxPool2D(pool_size=(2, 2), strides=(1, 1)),

        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        LeakyReLU(alpha=0.1),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),

        Dense(units=512),
        LeakyReLU(alpha=0.1),
        Dense(units=512),
        LeakyReLU(alpha=0.1),
        Dense(units=10, activation='softmax')

    ])

    return model
