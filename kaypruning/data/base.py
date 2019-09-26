import tensorflow as tf
import tensorflow_datasets as tfds

from kaypruning.configs import *
from kaypruning.utils import describe, glogger


class DataBunch:
    r"""
    DataBunch Class is used to load the dataset,
    pre-processing and dividing the data to train and test data
    which will be used inside the trainer class
    """

    @classmethod
    def _convert_images(cls, dataset):
        r"""
        Converting the images and flatting them
        :param dataset: raw dataset of images and labels
        :return: new mapped dataset with flattened images
        """

        def flatten(image):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image_reshaped = tf.reshape(image, [model_hparams.num_features])
            return tf.cast(image_reshaped, tf.float32)

        return dataset.map(
            lambda image, label: (
                flatten(image),
                label
            )
        )

    def __init__(self, name: str = 'mnist', batch: int = 32,
                 cache: bool = True, split=None):
        self.name = name
        ds = tfds.load(name=name, as_supervised=True, split=split)

        self.train = DataBunch._convert_images(ds['train'])
        self.test = DataBunch._convert_images(ds['test'])

        if cache:
            # speed things up considerably
            self.train = self.train.cache()
            self.test = self.test.cache()

        self.batch = batch

        glogger.info(describe(self))

    def get_train(self):
        return (self.train
                .shuffle(data_configs.shuffle)
                .batch(self.batch)
                .prefetch(data_configs.prefetch)
                .repeat(data_configs.repeat))

    def get_test(self):
        return (self.test
                .batch(self.batch)
                .prefetch(data_configs.prefetch)
                .repeat(data_configs.repeat))
