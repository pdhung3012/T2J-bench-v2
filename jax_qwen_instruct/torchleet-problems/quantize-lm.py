import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

class LSTM(nn.Module):
    @nn.compact
    def build(self):
        self.lstm_cell = nn.LSTMCell()
        self.dense = nn.Dense(features=1)

    def __call__(self, x):
        h_t, c_t = jnp.zeros((x.shape[0], self.lstm_cell.output_size)), jnp.zeros((x.shape[0], self.lstm_cell.output_size))
        for t in range(x.shape[1]):
            h_t, c_t = self.lstm_cell(x[:, t:t+1], (h_t, c_t))
        out = self.dense(h_t)
        return out

def process_sequence(inputs, prng_key):
    def step(carry, input_data):
        h_t, c_t = carry
        h_t, c_t = jnp.tanh(jnp.dot(input_data, jnp.eye(inputs.shape[-1]))), c_t
        return (h_t, c_t), h_t

    initial_carry = (jnp.zeros((inputs.shape[0],)), jnp.zeros((inputs.shape[0],)))
    _, outputs = jax.lax.scan(step, initial_carry, inputs)
    return outputs

def loss_fn(params, X, y):
    return jnp.mean((X - y) ** 2)

def main():
    batch_size = 32
    input_size = 10
    num_epochs = 100
    key = jax.random.PRNGKey(0)

    X_train = jax.random.normal(key, (batch_size, input_size))
    y_train = jax.random.normal(key, (batch_size, input_size))

    model = LSTM()
    params = model.init(key, X_train)
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        outputs = process_sequence(X_train, subkey)
        current_loss = loss_fn(params, outputs, y_train)

        grad = jax.grad(loss_fn)(params, outputs, y_train)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {current_loss:.4f}")

if __name__ == "__main__":
    main()
