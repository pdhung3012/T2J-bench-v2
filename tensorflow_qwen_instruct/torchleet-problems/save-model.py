import numpy as np  # MODIFIED: Consistently import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.optimizers import Adam
import tensorflow as tf

class SimpleModel(tf.keras.Model):
    """A simple neural network model using TensorFlow."""
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense = Dense(units=1)  # A layer with one output feature

    def call(self, inputs):
        """Forward pass of the model."""
        return self.dense(inputs)

def train_model(X, y):
    """Train the model with the given data."""
    model = SimpleModel()
    params = model(X)  # Initial parameters are the weights of the dense layer
    # Loss function and optimization setup
    loss_fn = lambda params: tf.reduce_mean((model(X) - y) ** 2)
    optimizer = Adam(learning_rate=0.001)
    
    for epoch in range(100):  # Simple training loop
        with tf.GradientTape() as tape:
            loss = loss_fn(params)
        gradients = tape.gradient(loss, params)
        optimizer.apply_gradients(zip(gradients, params))
    
    return params

def main():
    """Main function to execute the training and evaluation of the model."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])  # Training data
    y_train = np.array([[0.0], [2.0], [4.0], [6.0]])  # Expected outputs

    # Train the model
    trained_params = train_model(X_train, y_train)

    # Verify the model works after loading
    X_test = np.array([[0.5], [1.0], [1.5]])  # Test data
    model = SimpleModel()  # Initialize model
    predictions = model(X_test)  # Get predictions
    print(f"Predictions after training: {predictions}")

if __name__ == "__main__":  # Entry point for the program
    main()  # Execute the main function
