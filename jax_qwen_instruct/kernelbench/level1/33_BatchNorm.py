import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, BatchNorm, Relu, Flatten, LogSoftmax, MaxPool, Conv, ScaleShift

class Model:
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        net = [
            Conv(32, (5, 5), padding='SAME'),
            Relu,
            MaxPool((2, 2)),
            Conv(64, (5, 5), padding='SAME'),
            Relu,
            MaxPool((2, 2)),
            Flatten,
            Dense(1024),
            BatchNorm,
            Relu,
            Dense(10),
            LogSoftmax
        ]
        self.params, self.layers = Model.build_net(net)

    @staticmethod
    def build_net(net):
        init, apply = Model.make_net(net)
        return init, apply

    @staticmethod
    def make_net(net):
        init_keys, apply_keys = jax.random.split(jax.random.PRNGKey(0))
        return Model.make_layers(init_keys, apply_keys, net)

    @staticmethod
    def make_layers(init_keys, apply_keys, net):
        params = {}
        for i, layer in enumerate(net):
            if isinstance(layer, (Dense, Conv)):
                params[f'W{i}'], params[f'b{i}'] = Model.init_params(init_keys, layer)
            elif isinstance(layer, BatchNorm):
                params[f'mean{i}'], params[f'std{i}'] = Model.init_params(init_keys, layer)
            elif isinstance(layer, (Relu, LogSoftmax)):
                pass
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
        apply_fn = Model.apply_fn(params, net)
        return init_fn, apply_fn

    @staticmethod
    def init_params(random_key, layer):
        if isinstance(layer, (Dense, Conv)):
            return jax.random.normal(random_key, layer.shape)
        elif isinstance(layer, BatchNorm):
            return jnp.zeros(layer.shape), jnp.ones(layer.shape)
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    @staticmethod
    def apply_fn(params, net):
        def apply(x, **kwargs):
            for i, layer in enumerate(net):
                if isinstance(layer, (Dense, Conv)):
                    x = layer(x, **params[f'{i}'])
                elif isinstance(layer, BatchNorm):
                    x = layer(x, **params[f'{i}'])
                elif isinstance(layer, (Relu, LogSoftmax)):
                    x = layer(x)
                else:
                    raise ValueError(f"Unsupported layer type: {type(layer)}")
            return x
        return apply

batch_size = 64
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
