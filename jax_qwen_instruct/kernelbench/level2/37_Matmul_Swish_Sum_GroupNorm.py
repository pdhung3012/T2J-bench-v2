import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, GroupNorm, Relu, Initializer, BatchNorm, ParametrizedLayer

class Model:
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.bias_shape = bias_shape
        
        net = [
            Dense(out_features, key=jax.random.PRNGKey(0), w_init=Initializer(jnp.ones)),
            Relu(),
            GroupNorm(num_groups=num_groups, axis=-1),
            BatchNorm(),
            Dense(sum(bias_shape), key=jax.random.PRNGKey(1), b_init=Initializer(jnp.zeros))
        ]
        
        self.params = ParametrizedLayer(net)
    
    @vmap
    def forward(self, x):
        x = self.params.apply({'params': self.params.init_params}, x)
        x = jnp.where(x > 0, x * jnp.sigmoid(x), x)  # Swish activation
        x = x + self.params.apply({'params': self.params.init_params}, jnp.zeros_like(x), mutable=True)[1]
        return x

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

get_inputs = lambda: [jnp.random.rand(batch_size, in_features)]
get_init_inputs = lambda: [in_features, out_features, num_groups, bias_shape]
