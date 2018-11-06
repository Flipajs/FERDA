import numpy as np


def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.

    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)

    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert type(projective) == np.ndarray
    assert projective.ndim == 1 or (projective.ndim == 2 and (projective.shape[0] == 4) or (projective.shape[0] == 3))
    if projective.ndim == 1:
        return (projective / projective[-1])[0:-1]
    else:
        return (projective / projective[-1, :])[0:-1, :]


def e2p(euclidean):
    """
    Convert 2d or 3d euclidean to projective coordinates.

    :param euclidean: projective coordinate(s)
    :type euclidean: numpy.ndarray, shape=(2 or 3, n)

    :return: projective coordinate(s)
    :rtype: numpy.ndarray, shape=(3 or 4, n)
    """
    assert type(euclidean) == np.ndarray
    assert euclidean.ndim == 1 or (euclidean.ndim == 2 and (euclidean.shape[0] == 3 or euclidean.shape[0] == 2))
    if euclidean.ndim == 1:
        return np.append(euclidean, 1)
    else:
        return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))


def column(vector):
    """
    Return column vector.
    :param vector: np.ndarray
    :return: column vector
    :rtype: np.ndarray, shape=(n, 1)
    """
    return vector.reshape((-1, 1))