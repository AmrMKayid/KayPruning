import numpy as np
import tensorflow as tf

from kaypruning.prune import register


@register
def unit(weight: tf.Variable, k: float) -> tf.Variable:
    r"""
    Method that is used to perform Unit/Neurons Pruning on layer's weight matrix.
    The pruning is performed by: setting the smallest (k%) of entire columns to zero
    in the weight matrix according to their L2-norm.

    => Find the norm for each column in the matrix -> column_norm
    => Get the sorted indices from the column norm array
    => Compute the threshold using (k%) to eliminate the columns
    => Setting all the columns less than the threshold to ZERO

    Example: Assuming k is 0.5
    >>> weight =array([[3, 4, 2, 1], [8, 7, 6, 9]])
    >>> column_norm = array([8.54400375, 8.06225775, 6.32455532, 9.05538514])
    >>> sorted_indices = array([2, 1, 0, 3])
    >>> threshold = 2
    >>> indices = array([2, 1])
    >>> tmp_w = array([[3, 0, 0, 1], [8, 0, 0, 9]])

    :param weight: The weight matrix
    :param k: The percentage of columns that will be pruned
    :return: New unit pruned weight matrix
    """
    tmp_w = weight.numpy()
    column_norm = np.linalg.norm(tmp_w, axis=0)  # Finding norm of each column
    sorted_indices = np.argsort(column_norm)
    threshold = int(k * len(sorted_indices))
    indices = sorted_indices[0:threshold]
    tmp_w[:, indices] = 0

    return weight.assign(tmp_w)
