from pathlib import Path

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from models.Net import net_self


LOG_DIR = Path("log-self")
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
WEIGHTS_PATH = LOG_DIR / "self.h5"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    model = net_self()
    model.build(input_shape=(None, 32, 32, 3))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=str(TENSORBOARD_DIR))]
    model.fit(
        train_images,
        train_labels,
        batch_size=32,
        epochs=50,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
    )

    model.save_weights(str(WEIGHTS_PATH))
    print(f"Saved weights to: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
