import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from flax import linen as nn
import optax

# Linear regression model definition
class LinearRegressionModel(nn.Module):
    input_dim: int

    def setup(self):
        self.w = self.param('w', jnp.zeros, (self.input_dim, 1))
        self.b = self.param('b', jnp.zeros, (1,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

# Loss function
def loss_fn(model, inputs, targets):
    predictions = model(inputs)
    return jnp.mean((predictions - targets) ** 2)

# Jitted gradient computation using vectorization
@jit
def compute_gradients(model, inputs, targets):
    return grad(loss_fn)(model, inputs, targets)

# Training function
def train_model(model, inputs, targets, num_epochs=1000, learning_rate=0.01):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model)

    for epoch in range(num_epochs):
        grads = compute_gradients(model, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = model.apply(updates)

        if epoch % 100 == 0:
            current_loss = loss_fn(model, inputs, targets)
            print(f"Epoch {epoch}, Loss: {current_loss}")

    return model

def main():
    # Generate synthetic data
    inputs = jnp.array([[1.0], [2.0], [3.0]])  # Input features
    targets = jnp.array([[2.0], [3.0], [4.0]])  # Target output

    # Initialize model
    model = LinearRegressionModel(input_dim=1)

    # Train the model
    trained_model = train_model(model, inputs, targets)

if __name__ == "__main__":
    main()
