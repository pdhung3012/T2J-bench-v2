import jax.numpy as jnp  # MODIFIED: Consistent import of jax.numpy as jnp
from jax import random

def generate_random_numbers(shape):
    """
    Generate random numbers following a normal distribution.

    Args:
        shape (tuple): The shape of the output array.

    Returns:
        jnp.ndarray: An array of random numbers of the specified shape.
    """
    key = random.PRNGKey(0)
    return random.normal(key, shape)  # Example method to generate random numbers

# Example usage of the generate_random_numbers function
def main():
    # Generate a 3x3 array of random numbers
    random_numbers = generate_random_numbers((3, 3))
    print("Generated Random Numbers:\n", random_numbers)

if __name__ == "__main__":
    main()  # Entry point of the program
