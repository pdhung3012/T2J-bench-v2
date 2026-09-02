import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp

tfd = tfp.distributions

class SimpleModel(Model):
    """A simple feedforward neural network model."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = layers.Dense(128)
        self.relu = layers.ReLU()
        self.dense2 = layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


def create_model():
    """Create an instance of the SimpleModel."""
    return SimpleModel()


def compute_loss(logits, labels):
    """Compute the binary cross-entropy loss."""
    sigmoid_logits = tf.sigmoid(logits)
    return tf.reduce_mean(tf.losses.binary_crossentropy(labels, sigmoid_logits, from_logits=False))


def accuracy(logits, labels):
    """Calculate the accuracy of the model predictions."""
    predictions = tf.round(tf.sigmoid(logits))
    return tf.reduce_mean(tf.cast(predictions == labels, tf.float32))


@tf.function
def train_step(optimizer, model, batch):
    """Perform a single training step."""
    def loss_fn(params):
        with tf.GradientTape() as tape:
            logits = model(batch['X'])
            loss_value = compute_loss(logits, batch['y'])
        gradients = tape.gradient(loss_value, params)
        optimizer.apply_gradients(zip(gradients, params))
        return loss_value
    
    gradients = tf.GradientTape().differentiable_wrt(optimizer.target)
    optimizer.apply_gradients(zip(gradients, optimizer.target.variables))
    return optimizer


def train_model(X, y, num_epochs, key):
    """Train the model on the provided data."""
    model = create_model()
    params = model(X)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    dataset_size = tf.shape(X)[0]
    
    for epoch in range(num_epochs):
        # Shuffle dataset
        perm = tf.random.shuffle(tf.range(dataset_size))
        X_shuffled = tf.gather(X, perm)
        y_shuffled = tf.gather(y, perm)
        
        for i in range(0, dataset_size, 32):
            batch = {
                'X': X_shuffled[i:i + 32],
                'y': y_shuffled[i:i + 32]
            }
            optimizer = train_step(optimizer, model, batch)
        
        # Example log after each epoch
        logits = model(X)
        train_acc = accuracy(logits, y)
        print(f"Epoch {epoch + 1}, Train Accuracy: {train_acc:.4f}")


def main():
    """Main entry point for the script."""
    # Example data generation with explicit PRNG key
    key = tf.random.Generator.from_seed(0)  # Initialize PRNG key
    X = tf.random.uniform((1000, 10))  # MODIFIED: Added explicit PRNG key
    y = tf.constant([0, 1] * 500)  # Sample labels

    num_epochs = 10
    train_model(X, y, num_epochs, key)  # MODIFIED: pass key to train_model


if __name__ == "__main__":
    main()
