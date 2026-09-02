import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import flax.linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt
import numpy as np

def load_cifar10(batch_size=64):
    ds = tfds.load('cifar10', split='train', as_supervised=True)
    
    def preprocess(image, label):
        image = jax.image.resize(image, (32, 32))
        image = image / 255.0
        return image, label
    
    ds = ds.map(preprocess)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(jax.device_count() * jax.local_device_count())
    
    return tfds.as_numpy(ds)

def main():
    try:
        batch_size = 64  # Example batch size, adjust as necessary
        cifar10_data = load_cifar10(batch_size)
        
        # Example of iterating through the dataset and displaying images
        for images, labels in cifar10_data:
            print(images.shape, labels.shape)  # Print shapes to show output
            break  # Remove break to process all batches
        
    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    main()
