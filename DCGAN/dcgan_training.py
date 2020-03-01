import cntk as C
import cntk.io.transforms as xforms
import cv2
import numpy as np
import os

from cntk.layers import BatchNormalization, Convolution2D, ConvolutionTranspose2D
from pandas import DataFrame

img_channel = 3
img_height = 256
img_width = 256
num_classes = 1

z_dim = 100

iteration = 50000
minibatch_size = 16
num_samples = 612


def create_reader(map_file, train):
    transforms = [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2),
                  xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear")]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        images=C.io.StreamDef(field="image", transforms=transforms),
        dummy=C.io.StreamDef(field="label", shape=num_classes))), randomize=train)


def dcgan_generator(h):
    with C.layers.default_options(init=C.normal(0.02), pad=True, bias=False, map_rank=1, use_cntk_engine=True):
        h = C.reshape(h, (-1, 1, 1))

        h = ConvolutionTranspose2D((4, 4), 1024, pad=False, strides=1, output_shape=(4, 4))(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = ConvolutionTranspose2D((5, 5), 512, strides=2, output_shape=(img_height // 32, img_width // 32))(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = ConvolutionTranspose2D((5, 5), 256, strides=2, output_shape=(img_height // 16, img_width // 16))(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = ConvolutionTranspose2D((5, 5), 128, strides=2, output_shape=(img_height // 8, img_width // 8))(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = ConvolutionTranspose2D((5, 5), 64, strides=2, output_shape=(img_height // 4, img_width // 4))(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = ConvolutionTranspose2D((5, 5), 32, strides=2, output_shape=(img_height // 2, img_width // 2))(h)
        h = BatchNormalization()(h)
        h = C.relu(h)

        h = ConvolutionTranspose2D((5, 5), 3, strides=2, bias=True, output_shape=(img_height, img_width))(h)
        h = C.tanh(h)

        return h


def dcgan_discriminator(h):
    with C.layers.default_options(init=C.normal(0.02), pad=True, bias=False, map_rank=1, use_cntk_engine=True):
        h = Convolution2D((3, 3), 32, strides=2, bias=True)(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 64, strides=2)(h)
        h = BatchNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 128, strides=2)(h)
        h = BatchNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 256, strides=2)(h)
        h = BatchNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 512, strides=2)(h)
        h = BatchNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 1024, strides=2)(h)
        h = BatchNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((4, 4), 1, activation=C.sigmoid, pad=False, bias=True, strides=1)(h)

        return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train_dcgan_map.txt", True)

    #
    # latent, input, generator, and discriminator
    #
    z = C.input_variable(shape=(z_dim,), dtype="float32", needs_gradient=True)
    x = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    x_real = (x - 127.5) / 127.5

    G_fake = dcgan_generator(z)
    D_real = dcgan_discriminator(x_real)
    D_fake = D_real.clone(method="share", substitutions={x_real.output: G_fake.output})

    #
    # loss function
    #
    G_loss = - C.log(D_fake)
    D_loss = - C.log(D_real) - C.log(1.0 - D_fake)

    #
    # optimizer and cyclical learning rate
    #
    G_learner = C.adam(G_fake.parameters, lr=1e-4, momentum=0.5, unit_gain=False,
                       gradient_clipping_with_truncation=True, gradient_clipping_threshold_per_sample=minibatch_size)
    D_learner = C.adam(D_real.parameters, lr=1e-4, momentum=0.5, unit_gain=False,
                       gradient_clipping_with_truncation=True, gradient_clipping_threshold_per_sample=minibatch_size)
    G_progress_printer = C.logging.ProgressPrinter(tag="Generator")
    D_progress_printer = C.logging.ProgressPrinter(tag="Discriminator")


    if not os.path.exists("./dcgan_image"):
        os.mkdir("./dcgan_image")
    
    G_trainer = C.Trainer(G_fake, (G_loss, None), [G_learner], [G_progress_printer])
    D_trainer = C.Trainer(D_real, (D_loss, None), [D_learner], [D_progress_printer])

    input_map = {x: train_reader.streams.images}

    #
    # train DCGAN
    #
    logging = {"step": [], "G_loss": [], "D_loss": []}
    for step in range(iteration):
        #
        # train discriminator
        #
        z_data = np.ascontiguousarray(np.random.normal(size=(minibatch_size, z_dim)), dtype="float32")
        x_data = train_reader.next_minibatch(minibatch_size, input_map=input_map)

        batch_input = {x: x_data[x].data, z: z_data}

        D_trainer.train_minibatch(batch_input)
        D_step_loss = D_trainer.previous_minibatch_loss_average
        
        #
        # train generator
        #
        z_data = np.ascontiguousarray(np.random.normal(size=(minibatch_size, z_dim)), dtype="float32")

        batch_input = {z: z_data}

        output = G_trainer.train_minibatch(batch_input, outputs=[G_fake])
        G_step_loss = G_trainer.previous_minibatch_loss_average

        #
        # save image
        #
        if step % 10 == 0:
            image = np.transpose(list(output[1].values())[0][0] / 2 + 0.5, (1, 2, 0)) * 255
        
            if not os.path.exists("./dcgan_image/step%d" % step):
                os.mkdir("./dcgan_image/step%d" % step)

            cv2.imwrite("./dcgan_image/step%d/fake.png" % step, image)

        #
        # loss and error logging
        #
        logging["step"].append(step + 1)
        logging["G_loss"].append(G_step_loss)
        logging["D_loss"].append(D_step_loss)

        G_trainer.summarize_training_progress()
        D_trainer.summarize_training_progress()

    #
    # save model and logging
    #
    G_fake.save("./dcgan_generator.model")
    D_real.save("./dcgan_discriminator.model")
    print("Saved model.")

    df = DataFrame(logging)
    df.to_csv("./dcgan.csv", index=False)
    print("Saved logging.")
    
