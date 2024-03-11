import numpy as np
from scipy.stats import betabinom, beta

def generate_beta_binomial_samples(alpha_grid, n, m, num_samples):
    """
    Generate samples from the Beta-Binomial distribution for a grid of alpha values.
    
    :param alpha_grid: Grid of alpha values to simulate.
    :param n: Number of trials in each sample.
    :param m: Number of samples to generate for each alpha pair.
    :param num_samples: Number of data points in each sample.
    :return: Dictionary of samples for each pair of alpha values.
    """
    samples_dict = {}

    for alpha_1 in alpha_grid:
        for alpha_2 in alpha_grid:
            # Generate parameters for the Beta distribution
            a, b = alpha_1, alpha_2

            # Generate Beta distribution samples
            p = beta.rvs(a, b, size=m)

            # Generate Beta-Binomial samples
            samples = [betabinom.rvs(n, p_val, 1-p_val, size=num_samples) for p_val in p]
            samples_dict[(alpha_1, alpha_2)] = samples

    return samples_dict

# Parameters
n = 20  # Number of trials in each sample
m = 20  # Number of samples for each alpha pair
num_samples = 20  # Number of data points in each sample
alpha_values = np.arange(1, 11)  # Grid of alpha values

# Generate samples
beta_binomial_samples = generate_beta_binomial_samples(alpha_values, n, m, num_samples)
print(len(beta_binomial_samples[(1,2)]))  # Displaying the samples

