from __future__ import unicode_literals
import numpy as np


def angle_absolute_error_direction_agnostic(angles_pred, angles_true, backend=np):
    """
    Compute direction agnostic error between predicted and true angle(s).

    :param angles_pred: predicted angles; array like, shape=(1, n) or (n, 1)
    :param angles_true: true angles; array like, shape=(1, n) or (n, 1)
    :param backend: numpy or keras.backend module
    :return: error angle(s) in range (0, 90); array like, shape=(1, n) or (n, 1)
    """
    val = backend.abs(angles_pred - angles_true) % 180
    return backend.minimum(val, 180 - val)


def angle_absolute_error(angles_pred, angles_true, backend=np):
    angles_pred %= 360
    angles_true %= 360
    return backend.minimum(
        backend.abs(angles_pred - angles_true),
        180 - backend.abs(angles_pred % 180 - angles_true % 180)
    )
