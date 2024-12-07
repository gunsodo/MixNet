import copy
import logging
import os
import time
from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from keras.backend import set_image_data_format
from keras.callbacks import CallbackList, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.metrics import SparseCategoricalCrossentropy
from keras.models import Model

from ._types import Config
from .callbacks import ReduceLROnPlateau, TimeHistory
from .gradients import GradientBlending
from .utils import compute_class_weight, log_dict, mean_dict

logger = logging.getLogger(__name__)


class Trainer:
    """Base trainer"""

    def __init__(self, config: Config):
        self.config = config
        self.model: Model = None

        self.setup()

    def setup(self):
        # Optimizer
        self.optimizer = self.config.optimizer.optimizer
        self.optimizer.lr = self.config.optimizer.lr

        # Loss
        self.loss = self.config.loss.loss_dict

        # Gradient blending
        self.gradient = GradientBlending(
            n=self.config.loss.n_losses,
            init_weights=self.config.loss.init_loss_weights,
            adaptive_masked=self.config.loss.adaptive_gradient,
        )

        # Monitoring
        self.best_loss_weights: List[float] = None
        self.best_epoch: int = 0

        # Callbacks
        cb_config = self.config.callback
        log_config = self.config.log
        os.makedirs(log_config, exist_ok=True)

        self.callbacks = CallbackList(callbacks=[
            CSVLogger(log_config.csv_dir),
            ModelCheckpoint(
                filepath=log_config.weights_dir,
                **cb_config.model_checkpoint.model_dump(),
            ),
            TimeHistory(log_config.time_log),
            ReduceLROnPlateau(
                lr=self.optimizer.lr,
                **cb_config.reduce_lr.model_dump(),
            ),
            EarlyStopping(**cb_config.early_stopping.model_dump(), ),
        ])

        # Data format
        set_image_data_format(self.config.data_format)

        # Seed
        np.random.seed(self.config.training.seed)
        tf.random.set_seed(self.config.training.seed)

    def load_weights(self, weight_dir: str):
        logger.info("Loading weights from %s", weight_dir)
        self.model.load_weights(weight_dir)

    @tf.function
    @abstractmethod
    def train_step(self, x, y, loss_weights=None):
        """
        Custom train_step function
        """
        pass

    @tf.function
    @abstractmethod
    def val_step(self, x, y, loss_weights=None):
        """
        Custom val_step function
        """
        pass

    @tf.function
    @abstractmethod
    def test_step(self, x, y, loss_weights=None):
        """
        Custom test_step function
        """
        pass

    @tf.function
    @abstractmethod
    def pred_step(self, x):
        """
        Custom pred_step function
        """
        pass

    @abstractmethod
    def build(self, load_weights: bool = False) -> Model:
        pass

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model = self.build()
        self._training(x=X_train, y=y_train, validation_data=(X_val, y_val))

    def _get_prediction_dict(self, y_test, y_pred) -> dict[str, np.ndarray]:
        y_pred_argm = np.argmax(y_pred, axis=1)
        return {
            "y_true": y_test,
            "y_pred": y_pred_argm,
            "y_pred_clf": y_pred,
        }

    def evaluate(self, X_test, y_test) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        self.model = self.build(load_weights=True)
        tic = time.time()
        evaluation, y_pred = self._testing(X_test, y_test)
        pred_time = time.time() - tic

        mem_usage = tf.config.experimental.get_memory_info("GPU:0")["current"]
        logger.info("Checking average current GPU memory usage: %s",
                    mem_usage)

        y_dict = self._get_prediction_dict(y_test, y_pred)
        logger.info("Classification report:")
        logger.info(classification_report(y_test, y_dict["y_pred"]))

        # TODO: Should move to utils
        f1_average = self.config.training.f1_average
        logger.info("F1-score is computed with average=%s", f1_average)
        f1 = f1_score(y_test, y_dict["y_pred"], average=f1_average)
        evaluation = {
            **evaluation,
            "f1_score": f1,
            "prediction_time": pred_time,
            "memory_usage": mem_usage,
            **dict(zip([f"w_{name}_loss" for name in self.config.loss.loss_names],
                       self.best_loss_weights))
        }

        return y_dict, evaluation

    def get_dataset(self, x, y, batch_size: int, shuffle: bool = True):
        dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)).batch(self.batch_size)
        if self.config.training.seed and shuffle:
            dataset = dataset.shuffle(len(x), seed=self.config.training.seed)
        return dataset.batch(batch_size=batch_size)

    def _training(self, x, y, validation_data):
        """
        Training loop
        """
        x_val, y_val = validation_data
        x = tuple(x)
        y = tuple(y)
        x_val = tuple(x_val)
        y_val = tuple(y_val)

        if self.config.training.class_balancing:
            class_weight = compute_class_weight(y)
            logger.info(
                "This iteration is taking into account of class weight: %s",
                class_weight,
            )
            self.loss["crossentropy"] = SparseCategoricalCrossentropy(
                class_weight=class_weight)

        train_batch_size = self.config.training.batch_size
        n_batches = np.ceil(len(x) / train_batch_size)
        val_batch_size = len(x_val) // n_batches

        train_dataset = self.get_dataset(x, y, batch_size=train_batch_size)
        val_dataset = self.get_dataset(x_val, y_val, batch_size=val_batch_size)

        if self.config.loss.use_adaptive_gradient:
            self.gradient.build(
                policy=self.config.loss.policy,
                batch_size=train_batch_size,
                valid_batch_size=val_batch_size,
            )

        self.callbacks.on_train_begin()
        loss_weights = self.config.loss.init_loss_weights
        loss_names = self.config.loss.loss_names

        for epoch in range(self.config.training.epochs):
            self.callbacks.on_epoch_begin(epoch=epoch)

            train_logs = []
            val_logs = []

            for step, ((x_batch_train, y_batch_train),
                       (x_batch_val, y_batch_val)) in enumerate(
                           zip(train_dataset, val_dataset)):
                self.callbacks.on_train_batch_begin(step)
                tmp_logs, _ = self.train_step(x_batch_train, y_batch_train,
                                              loss_weights)
                train_logs.append(tmp_logs)
                self.callbacks.on_train_batch_end(step, tmp_logs)

                tmp_val_logs, _ = self.val_step(x_batch_val, y_batch_val,
                                                loss_weights)

                if self.config.loss.use_adaptive_gradient:
                    train_losses = [
                        tmp_logs[f"train_{name}_loss"].numpy()
                        for name in loss_names
                    ]
                    val_losses = [
                        tmp_val_logs[f"val_{name}_loss"].numpy()
                        for name in loss_names
                    ]
                    self.gradient.add_point_train_loss(*train_losses)
                    self.gradient.add_point_valid_loss(*val_losses)

                    if epoch >= self.config.loss.warmup_epoch:
                        new_loss_weights = self.gradient.compute_adaptive_weight(
                            to_tensor=True)
                        loss_weights = copy.copy(new_loss_weights)

                tmp_val_logs.update({
                    "w_" + name: w
                    for name, w in zip(loss_names, loss_weights)
                })
                val_logs.append(tmp_val_logs)

            epoch_logs = {**mean_dict(train_logs), **mean_dict(val_logs)}
            self.callbacks.on_epoch_end(epoch=epoch, logs=epoch_logs)
            # self.train_acc_metric.reset_states()
            # self.val_acc_metric.reset_states()

        self.callbacks.on_train_end()
        self.best_loss_weights = copy.copy(loss_weights).numpy()
        self.loss_weights = loss_weights

    def _testing(self, x, y) -> Tuple[Dict[str, np.ndarray], Tuple[np.ndarray]]:
        x = tuple(x)
        y = tuple(y)

        # Single batch inference
        test_dataset = self.get_dataset(x, y, batch_size=len(x), shuffle=False)

        x_batch_test, y_batch_test = next(test_dataset)
        test_logs, y_pred = self.test_step(
            x_batch_test, y_batch_test, self.loss_weights)
        test_logs = {k: v.numpy() for k, v in test_logs.items()}
        # self.test_acc_metric.reset_states()
        log_dict(test_logs)
        return test_logs, tuple(yp.numpy() for yp in y_pred)

    def prediction(self, x) -> Tuple[np.ndarray]:
        test_dataset = tf.data.Dataset.from_tensor_slices(
            tuple(x)).batch(len(x))
        y_pred = self.pred_step(next(test_dataset))
        # self.pred_acc_metric.reset_states()
        return tuple(yp.numpy() for yp in y_pred)
