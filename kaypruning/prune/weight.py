import numpy as np
import tensorflow as tf

from kaypruning.prune import register


@register
def weight(weight: tf.Variable, k: float) -> tf.Variable:
    r"""
    Method that is used to perform Weight Pruning on layer's weight matrix.
    The pruning is performed by: setting the smallest k% of individual weights
    in the weight matrix to zero according to their magnitude (absolute value).

    => Find the absolute for each weight in the matrix and sort them -> tmp_w
    => Compute the threshold using (k%) and get the element using the threshold_index
    => Compute the mask for which the values are higher than the threshold
    => Setting all the elemnt less than the threshold to ZERO using the mask

    Example: Assuming k is 0.5
    >>> w = array([[ 3, -6,  9,  1], [ 8,  2, -7,  4]])
    >>> tmp_w = array([1, 2, 3, 4, 6, 7, 8, 9])
    >>> threshold_index = 4
    >>> threshold = 6
    >>> mask = array([[False,  True,  True, False], [ True, False,  True, False]])
    >>> weight = array([[ 0, -6,  9,  0], [ 8,  0, -7,  0]]

    :param weight: The weight matrix
    :param k: The percentage of columns that will be pruned
    :return: New weighted pruned weight matrix
    """
    tmp_w = weight.numpy()
    tmp_w = np.sort(np.abs(np.reshape(tmp_w, [-1])))

    threshold_index = int(k * len(tmp_w))
    threshold = tmp_w[threshold_index]
    mask = ((weight >= threshold) | (weight <= -threshold))

    weight.assign(tf.reshape(
        tf.where(mask, weight, tf.zeros_like(weight)),
        tf.shape(weight))
    )

    return weight
