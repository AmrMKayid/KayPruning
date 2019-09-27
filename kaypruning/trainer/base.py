import os

import tensorflow as tf
from tensorflow.keras.models import Model

from kaypruning.configs import *
from kaypruning.data import DataBunch
from kaypruning.prune import get_prune_fn
from kaypruning.utils import glogger, describe


class Trainer:
    r"""
    Trainer class is used to
    => train the model
    => prune the model
    => save and restore the model
    => store all the metrics of the training process
    """

    def __init__(self, model: Model, db: DataBunch,
                 epochs: int = 5, path: str = training_hparams.path):
        self.model = model
        self.db = db
        self.epochs = epochs

        self.path = path + self.db.name
        os.makedirs(self.path, exist_ok=True)

        self.loss_object = tf.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(lr=training_hparams.lr)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.path, max_to_keep=3)

        self.metrics_list()

        glogger.info(describe(self))

        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            glogger.info("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            glogger.info("Initializing from scratch.")

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def train(self):
        for images, labels in self.db.get_train():
            self.train_step(images, labels)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def test(self):
        for images, labels in self.db.get_test():
            self.test_step(images, labels)

    def save(self):
        self.ckpt.step.assign_add(1)
        save_path = self.manager.save()
        glogger.info(f"Saved checkpoint for epoch {int(self.ckpt.step)}: {save_path}")

    def restore(self):
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            glogger.info(f"Restored from {self.manager.latest_checkpoint}")

    def reset(self):
        """
        Reset the metrics for the next epoch
        :return:
        """
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    def prune(self, prune_type, k):
        glogger.info(f"{prune_type} pruning: {k * 100}%")
        weights = self.model.trainable_weights[:-1]
        for i, w in enumerate(weights):
            new_pruned_weight = get_prune_fn(prune_type, w, k)
            # self.model.layers[i].set_weights(new_pruned_weight)
            self.model.layers[i].kernel.assign(new_pruned_weight)

    def test_prune(self):
        self.test()
        glogger.info(f"Test Loss: {self.test_loss.result()}, Test Accuracy: {self.test_accuracy.result() * 100} %\n\n")

    def run(self):
        for epoch in range(self.epochs):
            self.train()
            self.test()
            glogger.info(f"Epoch {epoch + 1}: ({self})")
            self.update_metrics()
            self.save()
            self.reset()

    def run_pruning(self):
        for prune_type in prune_configs.type:
            for percentage_k in prune_configs.k:
                self.restore()
                self.prune(prune_type, percentage_k)
                self.test_prune()
                self.update_metrics(prune=True, prune_type=prune_type)
                self.reset()

    def metrics_list(self):
        self.metrics = DotDict({
            'train': DotDict({
                'losses': [],
                'accuracies': []
            }),
            'test': DotDict({
                'losses': [],
                'accuracies': []
            }),
            'weight': DotDict({
                'losses': [],
                'accuracies': []
            }),
            'unit': DotDict({
                'losses': [],
                'accuracies': []
            }),
        })

    def update_metrics(self, prune=False, prune_type='weight'):
        if prune:
            self.metrics[prune_type].losses.append(self.test_loss.result())
            self.metrics[prune_type].accuracies.append(self.test_accuracy.result() * 100)
        else:
            self.metrics.train.losses.append(self.train_loss.result())
            self.metrics.train.accuracies.append(self.train_accuracy.result() * 100)
            self.metrics.test.losses.append(self.test_loss.result())
            self.metrics.test.accuracies.append(self.test_accuracy.result() * 100)

    def print_metrics(self):
        for i, k in enumerate(prune_configs.k):
            glogger.info(f"({i}) Sparsity (k): {k * 100}% =>"
                         f" Weight Loss: {self.metrics['weight'].losses[i]},"
                         f" Weight Accuracy: {self.metrics['weight'].accuracies[i]}%"
                         f" Unit Loss: {self.metrics['unit'].losses[i]},"
                         f" Unit Accuracy: {self.metrics['unit'].accuracies[i]}%")

    def get_metrics(self):
        return self.metrics['weight'].losses, self.metrics['weight'].accuracies, \
               self.metrics['unit'].losses, self.metrics['unit'].accuracies

    def __str__(self):
        return f"Train Loss: {self.train_loss.result()}, Accuracy: {self.train_accuracy.result() * 100}%," \
            f" Test Loss: {self.test_loss.result()}, Test Accuracy: {self.test_accuracy.result() * 100}%"
