import tensorflow as tf
from tensorflow.random import set_seed
from tensorflow.random import normal

def generate_random_numbers(shape):
    """
    Generate random numbers following a normal distribution.

    Args:
        shape (tuple): The shape of the output array.

    Returns:
        tf.Tensor: An array of random numbers of the specified shape.
    """
    set_seed(0)  # Seed for reproducibility
    return normal(shape=shape, mean=0.0, stddev=1.0)  # Example method to generate random numbers

# Example usage of the generate_random_numbers function
def main():
    # Generate a 3x3 array of random numbers
    random_numbers = generate_random_numbers((3, 3))
    print("Generated Random Numbers:\n", random_numbers)

if __name__ == "__main__":
    main()  # Entry point of the program
