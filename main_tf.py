import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np


def train(device='/gpu:0'):
    """ Based on this tutorial: https://www.tensorflow.org/tutorials/images/cnn"""
    with tf.device(device):
        (train_images, train_labels), (test_images,
                                       test_labels) = datasets.cifar10.load_data()

        train_images, test_images = train_images / 255.0, test_images / 255.0

        model = models.Sequential()
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(train_images, train_labels, epochs=2,
                            validation_data=(test_images, test_labels))


if __name__ == "__main__":

    print("########## TensorFlow version:", tf.__version__)
    print('#######################################################')
    print("########## Visible devices:", tf.config.get_visible_devices())
    print('#######################################################')

    print('########## CUDA run:')
    train('/gpu:0')

    print('########## CPU run:')
    train('/cpu:0')

    print('########## If everything worked correctly, CUDA run should be 2-3x faster')
