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
    preds = params @ x.T  # Using matrix multiplication for dot product
    return jnp.mean((preds - y) ** 2)

# Update function using functional programming
@jit
def update(params, x, y, learning_rate=0.1):
    grads = grad(loss_fn)(params, x, y)
    return params - learning_rate * grads

# Training function
def train_model(key, model, x, y, epochs=100):
    params = jnp.array([model.w, model.b])
    for _ in range(epochs):  # MODIFIED: Removed 'epoch' variable
        params = update(params, x, y)
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

    # Reshape parameters back to w and b
    w, b = params
    model.w = w.item()
    model.b = b.item()

    # Test the model
    predictions = model(x)
    print(f"Predictions for {x.tolist()}: {predictions.tolist()}")
    print(f"Trained weights: {model.w}, bias: {model.b}")

if __name__ == "__main__":
    main()
