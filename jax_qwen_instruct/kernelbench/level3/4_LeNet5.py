import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    @nn.compact
    def __init__(self, num_classes, **kwargs):
        super(Model, self).__init__(**kwargs)
        
        # Convolutional layers
        self.conv1 = nn.Conv(features=6, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv(features=16, kernel_size=(5, 5), stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Dense(features=120)
        self.fc2 = nn.Dense(features=84)
        self.fc3 = nn.Dense(features=num_classes)
    
    def __call__(self, x):
        # First convolutional layer with ReLU activation and max pooling
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.max_pool(x, (2, 2), (2, 2), 'SAME')
        
        # Second convolutional layer with ReLU activation and max pooling
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.max_pool(x, (2, 2), (2, 2), 'SAME')
        
        # Flatten the output for the fully connected layers
        x = jnp.reshape(x, (-1, 16 * 5 * 5))
        
        # First fully connected layer with ReLU activation
        x = jax.nn.relu(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = jax.nn.relu(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model (larger batch & image)
batch_size = 4096
num_classes = 20

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, 1, 32, 32))]

def get_init_inputs():
    return [num_classes]
