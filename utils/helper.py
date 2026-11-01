import numpy as np


def naming_conversion(lr_exponent: float):
    if lr_exponent == int(lr_exponent):
        return str(int(lr_exponent))
    else:
        return f"{lr_exponent:.1f}".replace('.', 'p')


def random_split(num_samples, seed: int=42):
    rng = np.random.default_rng(seed=seed)
    split_point = int(num_samples * 0.8)

    indices = np.arange(num_samples)
    shuffled_indices = rng.permutation(indices)

    return shuffled_indices[:split_point], shuffled_indices[split_point:]

