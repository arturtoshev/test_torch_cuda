""" 
Based on this tutorial:
https://coderzcolumn.com/tutorials/artifical-intelligence/haiku-cnn
and the data management from:
https://github.com/google/jax/blob/main/examples/datasets.py
"""


import array
import gzip
import os
from os import path
import struct
import urllib.request

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp

import time


# TODO: tell the Torch and TF dataloader to use the same dir.
_DATA = "data/"


def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urllib.request.urlretrieve(url, out_file)
    print(f"downloaded {url} to {_DATA}")


def _partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
#   train_labels = _one_hot(train_labels, 10)
#   test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels

class CNN(hk.Module):
    def __init__(self):
        super().__init__(name="CNN")
        self.conv_model = hk.Sequential([
            hk.Conv2D(output_channels=32, kernel_shape=(3, 3), padding="SAME"),
            jax.nn.relu,

            hk.Conv2D(output_channels=16, kernel_shape=(3, 3), padding="SAME"),
            jax.nn.relu,

            hk.Flatten(),
            hk.Linear(10),
            jax.nn.softmax])

    def __call__(self, x_batch):
        return self.conv_model(x_batch)


def train(device='gpu'):

    _device = jax.devices(device)[0]

    X_train, Y_train, _, _ = mnist()
    X_train, Y_train = jnp.array(X_train, dtype=jnp.float32), jnp.array(
        Y_train, dtype=jnp.float32)
    X_train = X_train.reshape(-1, 28, 28, 1)/255.0  # reshape and rescale
    # train only on the first 10k images to save time. If you want to change that, you will have to
    # take care of the last batch, if it is of different size than the others. Thats why we choose 10240 = 256*40
    X_train, Y_train = X_train[:10240], Y_train[:10240]
    X_train, Y_train = jax.device_put(
        X_train, _device), jax.device_put(Y_train, _device)

    classes = jnp.unique(Y_train)

    def CrossEntropyLoss(weights, input_data, actual):
        preds = conv_net.apply(weights, rng, input_data)
        one_hot_actual = jax.nn.one_hot(actual, num_classes=len(classes))
        log_preds = jnp.log(preds)
        return - jnp.sum(one_hot_actual * log_preds)

    def UpdateWeights(weights, gradients):
        return weights - learning_rate * gradients

    value_and_grad_CE = jax.jit(
        jax.value_and_grad(CrossEntropyLoss), device=_device)
    update_weights = jax.jit(UpdateWeights, device=_device)

    # Reproducibility ## Initializes model with same weights each time.
    rng = jax.random.PRNGKey(42)
    conv_net = hk.transform(lambda x: CNN()(x))
    params = conv_net.init(rng, X_train[:5])
    epochs = 2
    batch_size = 256
    learning_rate = jnp.array(1/1e4)
    batches = jnp.arange((X_train.shape[0]//batch_size))  # Batch Indices

    start_time = time.time()
    for epoch in range(epochs):
        losses = []  # Record loss of each batch
        acc = []
        for batch in batches:
            start, end = int(batch*batch_size), int(batch *
                                                    batch_size+batch_size)
            # Single batch of data
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            loss, param_grads = value_and_grad_CE(params, X_batch, Y_batch)
            params = jax.tree_map(update_weights, params,
                                  param_grads)  # Update Params
            pred = conv_net.apply(params, rng, X_batch)
            acc.append(jnp.argmax(pred, axis=-1) == Y_batch)  # Record Accuracy
            losses.append(loss)  # Record Loss

        print(f'Epoch {epoch+1}/2, loss: {jnp.array(losses).mean():.3f} '
              f'- accuracy: {jnp.concatenate(acc).mean():.3f}')

    print('Training time %.2f sec' % (time.time()-start_time))


if __name__ == "__main__":

    print("########## JAX version:", jax.__version__)
    print("########## Haiku version:", hk.__version__)
    print('#######################################################')
    print("########## Visible devices:", jax.devices())
    print('#######################################################')

    print('########## CUDA run:')
    train('gpu')

    print('########## CPU run:')
    train('cpu')

    print('########## If everything worked correctly, CUDA run should be 3-4x faster')
