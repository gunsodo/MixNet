import logging
import tensorflow as tf

from keras import layers
from keras.constraints import max_norm
from keras.models import Model

from mixnet.trainer import Trainer
from mixnet._types import Config
from mixnet.loss import MeanSquaredError, SparseCategoricalCrossentropy, triplet_loss


logger = logging.getLogger(__name__)


class MixNet(Trainer):
    """MixNet trainer"""

    def __init__(
        self,
        config: Config,
        input_shape=(1, 400, 20),
        latent_dim=None,
    ):
        super().__init__(config=config)
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.config.loss.loss = [
            MeanSquaredError(), triplet_loss(), SparseCategoricalCrossentropy()]
        self.config.loss.loss_names = ["mse", "triplet", "crossentropy"]
        self.config.loss.weights = [1.0, 1.0, 1.0]
        self.config.data_format = "channels_last"

    def build(self, load_weights=False) -> Model:
        """
        build a model and return as tf.keras.models.Model
        """
        self.D, self.T, self.C = self.input_shape

        if self.latent_dim is not None:
            self.LATENT_DIM = self.latent_dim
        else:
            if self.config.training.num_class == 2:
                self.LATENT_DIM = self.C
            else:
                self.LATENT_DIM = 64

        self.sfreq = 100
        self.P1 = (1, self.T//self.sfreq)
        self.P2 = (1, 4)  # MI Task
        self.F1 = self.C
        self.F2 = self.C//2
        self.FLAT = self.T//self.P1[1]//self.P2[1]

        # encoder
        encoder_input = layers.Input(self.input_shape, name="en_input")
        en_conv = layers.Conv2D(self.F1, (1, 64), activation="elu", padding="same",
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)), name="en_conv1")(encoder_input)
        en_conv = layers.BatchNormalization(
            axis=3, epsilon=1e-05, momentum=0.1, name="en_bn1")(en_conv)
        en_conv = layers.AveragePooling2D(
            pool_size=self.P1, name="en_avg1")(en_conv)
        en_conv = layers.Conv2D(self.F2, (1, 32), activation="elu", padding="same",
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)), name="en_conv2")(en_conv)
        en_conv = layers.BatchNormalization(
            axis=3, epsilon=1e-05, momentum=0.1, name="en_bn2")(en_conv)
        en_conv = layers.AveragePooling2D(
            pool_size=self.P2, name="en_avg2")(en_conv)
        en_conv = layers.Flatten(name="en_flat")(en_conv)
        z = layers.Dense(self.LATENT_DIM, kernel_constraint=max_norm(
            0.5), name="z")(en_conv)
        encoder = Model(inputs=encoder_input, outputs=z, name="encoder")
        encoder.summary()

        # decoder
        decoder_input = layers.Input(shape=(self.LATENT_DIM,), name="de_input")
        de_conv = layers.Dense(1*self.FLAT*self.F2, activation="elu",
                               kernel_constraint=max_norm(0.5), name="de_dense")(decoder_input)
        de_conv = layers.Reshape(
            (1, self.FLAT, self.F2), name="de_reshape")(de_conv)
        de_conv = layers.Conv2DTranspose(filters=self.F2, kernel_size=(1, 64),
                                         activation="elu", padding="same", strides=self.P2,
                                         kernel_constraint=max_norm(2., axis=(0, 1, 2)), name="de_deconv1")(de_conv)
        decoder_output = layers.Conv2DTranspose(filters=self.F1, kernel_size=(1, 32),
                                                activation="elu", padding="same", strides=self.P1,
                                                kernel_constraint=max_norm(2., axis=(0, 1, 2)), name="de_deconv2")(de_conv)
        decoder = Model(inputs=decoder_input,
                        outputs=decoder_output, name="decoder")
        decoder.summary()

        # Build the computation graph for training
        z = encoder(encoder_input)
        xr = decoder(z)
        y = layers.Dense(self.config.training.num_class, activation="softmax",
                         kernel_constraint=max_norm(0.5), name="classifier")(z)
        model = Model(inputs=encoder_input, outputs=[
                      xr, z, y], name=self.__class__.__name__)
        model.summary()

        if load_weights:
            logger.info("Loading weights from %s", self.config.log.weights_dir)
            model.load_weights(self.config.log.weights_dir)

        return model

    @tf.function
    def train_step(self, x, y, loss_weights):
        with tf.GradientTape() as tape:
            xr, z, y_logis = self.model(x, training=True)
            mse_loss = self.loss["mse"](x, xr)
            trp_loss = self.loss["triplet"](y, z)
            crossentropy_loss = self.loss["crossentropy"](y, y_logis)
            losses = [mse_loss, trp_loss, crossentropy_loss]
            train_loss = tf.reduce_sum(loss_weights * losses)
            self.config.metric.train.update_state(y, y_logis)

        grads = tape.gradient(train_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))
        logs = {
            "train_loss": train_loss,
            **dict(zip([f"train_{loss_name}_loss" for loss_name in self.config.loss.loss_names], losses)),
            "train_acc": self.config.metric.train.result(),
        }
        return logs, (xr, z, y_logis)

    @tf.function
    def val_step(self, x, y, loss_weights):
        xr, z, y_logis = self.model(x, training=False)
        mse_loss = self.loss["mse"](x, xr)
        trp_loss = self.loss["triplet"](y, z)
        crossentropy_loss = self.loss["crossentropy"](y, y_logis)
        losses = [mse_loss, trp_loss, crossentropy_loss]
        val_loss = tf.reduce_sum(loss_weights * losses)

        self.config.metric.val.update_state(y, y_logis)
        logs = {
            "val_loss": val_loss,
            **dict(zip([f"val_{loss_name}_loss" for loss_name in self.config.loss.loss_names], losses)),
            "val_acc": self.config.metric.val.result(),
        }
        return logs, (xr, z, y_logis)

    # TODO: Merge with above
    @tf.function
    def test_step(self, x, y, loss_weights):
        xr, z, y_logis = self.model(x, training=False)
        mse_loss = self.loss["mse"](x, xr)
        trp_loss = self.loss["triplet"](y, z)
        crossentropy_loss = self.loss["crossentropy"](y, y_logis)
        losses = [mse_loss, trp_loss, crossentropy_loss]
        test_loss = tf.reduce_sum(loss_weights * losses)

        self.config.metric.test.update_state(y, y_logis)
        logs = {
            "test_loss": test_loss,
            **dict(zip([f"test_{loss_name}_loss" for loss_name in self.config.loss.loss_names], losses)),
            "test_acc": self.config.metric.test.result(),
        }
        return logs, (xr, z, y_logis)

    @tf.function
    def pred_step(self, x):
        xr, z, y_logis = self.model(x, training=False)
        return (xr, z, y_logis)
