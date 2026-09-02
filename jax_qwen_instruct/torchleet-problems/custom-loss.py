import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
import optax

# Define a simple model
class LinearModel:
    def __init__(self, key):
        self.w = random.normal(key, ())
        self.b = random.normal(key, ())

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

# Loss function
def loss_fn(params, x, y):
    preds = params[0] * x + params[1]
    return jnp.mean((preds - y) ** 2)

# Update function using functional programming
def update(params, x, y, learning_rate=0.1):
    loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params = params - learning_rate * grads
    return params

# Training function
def train_model(key, model, x, y, epochs=100):
    params = (model.w, model.b)
    for _ in range(epochs):  # MODIFIED: Removed epoch variable
        params = update(params, x, y)  # MODIFIED: Updated update function call
    return params

def main():
    # Generate synthetic data
    key = random.PRNGKey(0)  # MODIFIED: Explicit PRNG key
    model = LinearModel(key)
    
    # Generate synthetic data
    x = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([[2.0], [4.0], [6.0]])

    # Train the model
    params = train_model(key, model, x, y, epochs=100)

    # Test the model
    predictions = params[0] * x + params[1]
    print(f"Predictions for {x.tolist()}: {predictions.tolist()}")
    print(f"Trained weights: {params[0]}, bias: {params[1]}")

if __name__ == "__main__":
    main()
