import tensorflow as tf
from tensorflow.random import set_seed  # MODIFIED: Import for setting seed

def init_params(key: tf.Tensor, input_shape: tuple) -> tf.Tensor:
    """Initialize parameters for the model."""
    param_shape = (input_shape[0], 1)  # Example shape for parameters
    return tf.random.normal(key, param_shape)  # Use explicit PRNG key

def loss_fn(params: tf.Tensor, inputs: tf.Tensor, targets: tf.Tensor) -> float:
    """Calculate the loss."""
    predictions = tf.tensordot(inputs, params, axes=[[1], [0]])  # Simulate predictions
    return tf.reduce_mean(tf.square(predictions - targets))  # Mean Squared Error

def main() -> None:
    """Main entry point for the program."""
    set_seed(0)  # Create an explicit PRNG key
    input_shape = (5, 10)  # Define input shape
    params = init_params(tf.constant([0.0]), input_shape)  # Initialize parameters
    inputs = tf.ones((5, 10))  # Example input data
    targets = tf.ones((5,))  # Example target data

    # Calculate loss
    loss_value = loss_fn(params, inputs, targets)  # Using loss function
    print(f"Loss: {loss_value.numpy()}")  # Displaying loss

if __name__ == "__main__":
    main()  # Entry point for the program
