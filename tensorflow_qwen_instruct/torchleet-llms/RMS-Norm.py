import tensorflow as tf
from tensorflow.keras.layers import Layer

class RMSNorm(Layer):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = self.add_weight(shape=(dim,), initializer='ones', trainable=True)  # gamma

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: shape (..., dim)
        norm = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)  # RMS
        return (x / norm) * self.scale

x = tf.random.normal((3, 5))  # e.g., (batch_size=3, features=5)
rmsnorm = RMSNorm(dim=5)
out = rmsnorm(x)
print(out.shape)  # should be (3, 5)
assert out.shape == (3, 5), "Output shape mismatch"
