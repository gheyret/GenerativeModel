import cntk as C
import cntk.io.transforms as xforms
import cv2
import numpy as np
import os
import pandas as pd

from cntk.layers import Convolution2D, ConvolutionTranspose2D
from cntk.layers.blocks import _INFERRED
from cntk.ops.functions import BlockFunction

img_channel = 3
img_height = 512
img_width = 512
num_classes = 1

epoch_size = 500
minibatch_size = 1
num_samples = 101

lambda_x = lambda_y = 10.0


def InstanceNormalization(shape, initial_scale=1, initial_bias=0, epsilon=C.default_override_or(0.00001), name=''):
    epsilon = C.get_default_override(InstanceNormalization, epsilon=epsilon)

    dtype = C.get_default_override(None, dtype=C.default_override_or(np.float32))

    scale = C.Parameter(shape, init=initial_scale, name='scale')
    bias = C.Parameter(shape, init=initial_bias, name='bias')
    epsilon = np.asarray(epsilon, dtype=dtype)

    @BlockFunction('InstanceNormalization', name)
    def instance_normalize(x):
        mean = C.reduce_mean(x, axis=(1, 2))
        x0 = x - mean
        std = C.sqrt(C.reduce_mean(x0 * x0, axis=(1, 2)))
        if epsilon != 0:
            std += epsilon
        x_hat = x0 / std
        return x_hat * scale + bias

    return instance_normalize


def create_reader(map_file, is_train):
    transforms = [xforms.crop(crop_type="center", side_ratio=0.9),
                  xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear")]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        image=C.io.StreamDef(field="image", transforms=transforms),
        dummy=C.io.StreamDef(field="label", shape=num_classes))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def residual_block(h, num_filters):
    with C.layers.default_options(init=C.normal(0.02), pad=True, strides=1, bias=False):
        h1 = C.relu(InstanceNormalization((num_filters, 1, 1))(Convolution2D((3, 3), num_filters)(h)))
        h2 = InstanceNormalization((num_filters, 1, 1))(Convolution2D((3, 3), num_filters)(h1))
        return C.relu(h2 + h)


def cyclegan_generator(h):
    with C.layers.default_options(init=C.normal(0.02), pad=True, strides=1, bias=False):
        h = C.relu(InstanceNormalization((64, 1, 1))(Convolution2D((7, 7), 64)(h)))
        h = C.relu(InstanceNormalization((128, 1, 1))(Convolution2D((3, 3), 128, strides=2)(h)))
        h = C.relu(InstanceNormalization((256, 1, 1))(Convolution2D((3, 3), 256, strides=2)(h)))

        h = residual_block(h, 256)
        h = residual_block(h, 256)
        h = residual_block(h, 256)

        h = residual_block(h, 256)
        h = residual_block(h, 256)
        h = residual_block(h, 256)

        h = residual_block(h, 256)
        h = residual_block(h, 256)
        h = residual_block(h, 256)

        h = C.relu(InstanceNormalization((128, 1, 1)))(
            ConvolutionTranspose2D((3, 3), 128, strides=2, output_shape=(img_height // 2, img_width // 2))(h)))
        h = C.relu(InstanceNormalization((64, 1, 1)))(
            ConvolutionTranspose2D((3, 3), 64, strides=2, output_shape=(img_height, img_width))(h)))
        h = Convolution2D((7, 7), 3, activation=C.tanh, bias=True)(h)

        return h


def cyclegan_discriminator(h):
    with C.layers.default_options(init=C.normal(0.02), pad=True, bias=False):
        h = C.leaky_relu(Convolution2D((3, 3), 64, strides=2, bias=True)(h), alpha=0.2)

        h = C.leaky_relu(InstanceNormalization((128, 1, 1)))(Convolution2D((3, 3), 128, strides=2)(h)), alpha=0.2)
        h = C.leaky_relu(InstanceNormalization((256, 1, 1)))(Convolution2D((3, 3), 256, strides=2)(h)), alpha=0.2)
        h = C.leaky_relu(InstanceNormalization((512, 1, 1)))(Convolution2D((3, 3), 512, strides=2)(h)), alpha=0.2)

        h = Convolution2D((1, 1), 1, activation=C.sigmoid, bias=True)(h)

        return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    x_train_reader = create_reader("./train_cyclegan_x_map.txt", is_train=True)
    y_train_reader = create_reader("./train_cyclegan_y_map.txt", is_train=True)

    #
    # generator
    #
    x = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    y = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)

    x_real = (x - 127.5) / 127.5
    y_real = (y - 127.5) / 127.5
    
    F_fake = cyclegan_generator(y_real)  # F(Y) -> X
    G_fake = cyclegan_generator(x_real)  # G(X) -> Y

    x_hat = F_fake.clone(method="share", substitutions={y_real.output: G_fake.output})  # F(G(X)) -> X'
    y_hat = G_fake.clone(method="share", substitutions={x_real.output: F_fake.output})  # G(F(Y)) -> Y'

    #
    # discriminator
    #
    Dx_real = cyclegan_discriminator(x_real)
    Dx_fake = Dx_real.clone(method="share", substitutions={x_real.output: F_fake.output})

    Dy_real = cyclegan_discriminator(y_real)
    Dy_fake = Dy_real.clone(method="share", substitutions={y_real.output: G_fake.output})

    #
    # loss function
    #
    cycle_consistency_loss = lambda_x * C.reduce_mean(C.abs(x_hat - x_real)) + \
                             lambda_y * C.reduce_mean(C.abs(y_hat - y_real))

    F_loss = C.reduce_mean(C.square(Dx_fake - 1.0)) / 2 + cycle_consistency_loss
    G_loss = C.reduce_mean(C.square(Dy_fake - 1.0)) / 2 + cycle_consistency_loss
    Dx_loss = C.reduce_mean(C.square(Dx_real - 1.0)) / 2 + C.reduce_mean(C.square(Dx_fake)) / 2
    Dy_loss = C.reduce_mean(C.square(Dy_real - 1.0)) / 2 + C.reduce_mean(C.square(Dy_fake)) / 2

    #
    # optimizer
    #
    F_learner = C.adam(F_fake.parameters, lr=1e-4, momentum=0.0,
                       gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    G_learner = C.adam(G_fake.parameters, lr=1e-4, momentum=0.0,
                       gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    Dx_learner = C.adam(Dx_real.parameters, lr=1e-4, momentum=0.5,
                        gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    Dy_learner = C.adam(Dy_real.parameters, lr=1e-4, momentum=0.5,
                        gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)

    F_progress_printer = C.logging.ProgressPrinter(tag="F Generator")
    G_progress_printer = C.logging.ProgressPrinter(tag="G Generator")
    Dx_progress_printer = C.logging.ProgressPrinter(tag="X Discriminator")
    Dy_progress_printer = C.logging.ProgressPrinter(tag="Y Discriminator")

    if not os.path.exists("./image"):
        os.makedirs("./image/F")
        os.makedirs("./image/G")

    if not os.path.exists("./tensorboard"):
        os.makedirs("./tensorboard/F")
        os.makedirs("./tensorboard/G")
        os.makedirs("./tensorboard/Dx")
        os.makedirs("./tensorboard/Dy")
    F_tensorabord_writer = C.logging.TensorBoardProgressWriter(freq=10, log_dir="./tensorboard/F", model=F_fake)
    G_tensorabord_writer = C.logging.TensorBoardProgressWriter(freq=10, log_dir="./tensorboard/G", model=G_fake)
    Dx_tensorabord_writer = C.logging.TensorBoardProgressWriter(freq=10, log_dir="./tensorboard/Dx", model=Dx_real)
    Dy_tensorabord_writer = C.logging.TensorBoardProgressWriter(freq=10, log_dir="./tensorboard/Dy", model=Dy_real)

    F_trainer = C.Trainer(F_fake, (F_loss, None), [F_learner], [F_progress_printer, F_tensorabord_writer])
    G_trainer = C.Trainer(G_fake, (G_loss, None), [G_learner], [G_progress_printer, G_tensorabord_writer])
    Dx_trainer = C.Trainer(Dx_real, (Dx_loss, None), [Dx_learner], [Dx_progress_printer, Dx_tensorabord_writer])
    Dy_trainer = C.Trainer(Dy_real, (Dy_loss, None), [Dy_learner], [Dy_progress_printer, Dy_tensorabord_writer])

    x_input_map = {x: x_train_reader.streams.image}
    y_input_map = {y: y_train_reader.streams.image}

    #
    # training
    #
    logging = {"epoch": [], "F_loss": [], "G_loss": [], "Dx_loss": [], "Dy_loss": []}
    for epoch in range(epoch_size):
        sample_count = 0
        F_epoch_loss = 0
        G_epoch_loss = 0
        Dx_epoch_loss = 0
        Dy_epoch_loss = 0
        while sample_count < num_samples:
            #
            # discriminator
            #
            x_real = x_train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=x_input_map)
            y_real = y_train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=y_input_map)

            x_fake = F_fake.eval(y_real)
            y_fake = G_fake.eval(x_real)

            Dx_trainer.train_minibatch({Dx_loss.arguments[0]: x_real[x].data, Dx_loss.arguments[1]: x_fake})
            Dy_trainer.train_minibatch({Dy_loss.arguments[0]: y_real[y].data, Dy_loss.arguments[1]: y_fake})

            Dx_epoch_loss += Dx_trainer.previous_minibatch_loss_average
            Dy_epoch_loss += Dy_trainer.previous_minibatch_loss_average
            
            #
            # generator
            #
            F_output = F_trainer.train_minibatch({F_loss.arguments[0]: y_real[y].data,
                                                  F_loss.arguments[1]: x_real[x].data}, outputs=[F_fake])
            G_output = G_trainer.train_minibatch({G_loss.arguments[0]: x_real[x].data,
                                                  G_loss.arguments[1]: y_real[y].data}, outputs=[G_fake])
            F_epoch_loss += F_trainer.previous_minibatch_loss_average
            G_epoch_loss += G_trainer.previous_minibatch_loss_average

            sample_count += minibatch_size

        #
        # tensorboard image
        #
        if epoch % 10 == 0:
            F_image = np.transpose(list(F_output[1].values())[0][0] / 2 + 0.5, (1, 2, 0)) * 255
            G_image = np.transpose(list(G_output[1].values())[0][0] / 2 + 0.5, (1, 2, 0)) * 255
            
            if not os.path.exists("./image/F/epoch%d" % epoch):
                os.makedirs("./image/F/epoch%d" % epoch)

            if not os.path.exists("./image/G/epoch%d" % epoch):
                os.makedirs("./image/G/epoch%d" % epoch)

            cv2.imwrite("./image/F/epoch%d/fake.png" % epoch, F_image)  # F(Y) -> X
            cv2.imwrite("./image/G/epoch%d/fake.png" % epoch, G_image)  # G(X) -> Y

        #
        # Dx loss, Dy loss, F loss and G loss logging
        #
        logging["epoch"].append(epoch + 1)
        logging["F_loss"].append(F_epoch_loss / (num_samples / minibatch_size))
        logging["G_loss"].append(G_epoch_loss / (num_samples / minibatch_size))
        logging["Dx_loss"].append(Dx_epoch_loss / (num_samples / minibatch_size))
        logging["Dy_loss"].append(Dy_epoch_loss / (num_samples / minibatch_size))

        F_trainer.summarize_training_progress()
        G_trainer.summarize_training_progress()
        Dx_trainer.summarize_training_progress()
        Dy_trainer.summarize_training_progress()

    #
    # save model and logging
    #
    F_fake.save("./cyclegan_F_generator.model")
    G_fake.save("./cyclegan_G_generator.model")
    Dx_real.save("./cyclegan_Dx_discriminator.model")
    Dy_real.save("./cyclegan_Dy_discriminator.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./cyclegan.csv", index=False)
    print("Saved logging.")
    
