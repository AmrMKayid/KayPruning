from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from kaypruning.configs import *
from kaypruning.models.registry import register
from kaypruning.utils import glogger, describe


class BaseLayer(Layer):

    def __init__(self, units):
        super(BaseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        """
        # Use build to create variables, as shape can be inferred from previous layers
        # If you were to create layers in __init__, one would have to provide input_shape
        # (same as it occurs in PyTorch for example)
        :param input_shape: the shape of the input to the layer
        :return:
        """
        self.kernel = self.add_weight(name="kernel",
                                      shape=[int(input_shape[-1]),
                                             self.units],
                                      initializer="random_normal",
                                      trainable=True,
                                      )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)


@register
class BaseModel(Model):
    r"""
    BaseModel that extend Keras Model API
    The model is dynamic and can have multiple layers
    according to the model_hparams.hidden_layers_size
    """

    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__(name='BaseModel')

        layers_size = model_hparams.hidden_layers_size
        for i, size in enumerate(layers_size):
            setattr(self, f'layer_{i + 1}', BaseLayer(size))
        self.classifier = BaseLayer(num_classes)

        glogger.info(describe(self))

    def call(self, input_tensor):
        x = self.layer_1(input_tensor)
        x = tf.nn.relu(x)

        layers = [l for l in vars(self) if l.startswith('layer')][1:]
        for layer in layers:
            x = getattr(self, layer)(x)
            x = tf.nn.relu(x)

        x = self.classifier(x)
        x = tf.nn.softmax(x)

        return x
