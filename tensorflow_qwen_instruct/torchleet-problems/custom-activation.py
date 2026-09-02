import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow as tf

# Initialize PRNG key
key = jax.random.PRNGKey(0)  # // MODIFIED: Initialize PRNG key explicitly

# Define the model function
def model(X, key):  # // MODIFIED: Pass PRNG key as a parameter
    w_key, b_key = jax.random.split(key)  # Split key for weights and bias
    w = jax.random.normal(w_key, ())  # // MODIFIED: Use PRNG key for randomness
    b = jax.random.normal(b_key, ())  # // MODIFIED: Use PRNG key for randomness
    return tf.tensordot(X, w, axes=1) + b  # // MODIFIED: Use tensordot instead of dot

# Jitted function to compute the loss
@tf.function  # // MODIFIED: Decorate with tf.function for compilation
def loss_fn(X, y, key):  # // MODIFIED: Pass PRNG key as a parameter
    pred = model(X, key)  # Use key here
    return tf.reduce_mean(tf.square(pred - y))  # // MODIFIED: Use reduce_mean and square

# Function to perform optimization step
@tf.function  # // MODIFIED: Ensure this function is stateless
def update(params, X, y, key):
    grads = tf.gradients(loss_fn(X, y, key), params)  # Compute gradients
    return params - 0.01 * grads[0]  # Simple SGD update

def main():
    # Data preparation
    X = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([[2.0], [4.0], [6.0]])

    # Convert to TensorFlow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # Model fitting
    params = None  # Initialize parameters (could be weights and bias)

    for epoch in range(100):  # Training loop
        params = update(params, X, y, key)  # // MODIFIED: Key passed in updates

    # Visualization
    plt.scatter(X.numpy(), y.numpy(), label='Data')
    plt.plot(X.numpy(), model(X, key).numpy(), 'r', label='Model Fit')  # // MODIFIED: Key used
    plt.legend()
    plt.show()

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(X_test, key)  # // MODIFIED: Pass key during prediction
    print(f"Predictions for {X_test.tolist()}: {predictions.numpy().tolist()}")

if __name__ == "__main__":
    main()
