import numpy as np


def grad_exp(
        theta: np.ndarray,
        x_s_a: np.ndarray
) -> np.ndarray:
    """
    Calculate the gradient of the exponential of theta * x_s_a.

    :param theta: Theta vector.
    :param x_s_a: State-action feature vector.
    :return: Gradient.
    """

    if theta.shape != x_s_a.shape:
        raise ValueError('Shape mismatch')

    return np.exp(theta.dot(x_s_a)) * x_s_a
