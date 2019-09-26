from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from kaypruning.configs import *
from kaypruning.models import register
from kaypruning.utils import glogger, describe


@register
class KerasModel(Model):
    def __init__(self, num_classes=10):
        super(KerasModel, self).__init__()
        layers_size = model_hparams.hidden_layers_size
        self.network = keras.Sequential(
            [Dense(size, activation=tf.nn.relu) for size in layers_size] +
            [Dense(num_classes, activation=tf.nn.softmax)]
        )

        glogger.info(describe(self))

    def call(self, input_tensor):
        return self.network(input_tensor)
