""" 
Based on this tutorial:
https://coderzcolumn.com/tutorials/artifical-intelligence/haiku-cnn
"""

import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow.keras import datasets

import time


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

    (X_train, Y_train), (X_test, Y_test) = datasets.fashion_mnist.load_data()
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
