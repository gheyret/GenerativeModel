import cntk as C
import cntk.io.transforms as xforms
import cv2
import numpy as np
import os

from cntk.layers import BatchNormalization, Convolution2D, ConvolutionTranspose2D, LayerNormalization
from pandas import DataFrame

img_channel = 3
img_height = 256
img_width = 256
num_classes = 1

z_dim = 100

n_critic = 5
iteration = 50000
minibatch_size = 16
num_samples = 612


def create_reader(map_file, train):
    transforms = [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2),
                  xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear")]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        image=C.io.StreamDef(field="image", transforms=transforms),
        dummy=C.io.StreamDef(field="label", shape=num_classes))), randomize=train)


def wgan_generator(h):
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


def wgan_critic(h):
    with C.layers.default_options(init=C.normal(0.02), pad=True, bias=False):
        h = Convolution2D((3, 3), 32, strides=2, bias=True)(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 64, strides=2)(h)
        h = LayerNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 128, strides=2)(h)
        h = LayerNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 256, strides=2)(h)
        h = LayerNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 512, strides=2)(h)
        h = LayerNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((3, 3), 1024, strides=2)(h)
        h = LayerNormalization()(h)
        h = C.leaky_relu(h, alpha=0.2)

        h = Convolution2D((4, 4), 1, pad=False, strides=1, bias=True)(h)

        return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train_wgan_map.txt", True)

    #
    # latent, input, generator, and critic
    #
    z = C.input_variable(shape=(z_dim,), dtype="float32", needs_gradient=True)
    x = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    x_real = (x - 127.5) / 127.5

    G_fake = wgan_generator(z)
    C_real = wgan_critic(x_real)
    C_fake = C_real.clone(method="share", substitutions={x_real.output: G_fake.output})

    #
    # gradient penalty
    #
    gradient_penalty = C.input_variable(shape=(1, 1, 1), dtype="float32")
    epsilon = C.random.uniform((1,), dtype="float32")
    interpolate = epsilon * x_real.output + (1 - epsilon) * G_fake.output

    #
    # loss function
    #
    G_loss = - C_fake
    C_loss = - C_real + C_fake + 10 * gradient_penalty

    #
    # optimizer
    #
    G_learner = C.adam(G_fake.parameters, lr=1e-4, momentum=0.0, unit_gain=False,
                       gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    C_learner = C.adam(C_real.parameters, lr=1e-4, momentum=0.0, unit_gain=False,
                       gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    G_progress_printer = C.logging.ProgressPrinter(tag="Generator")
    C_progress_printer = C.logging.ProgressPrinter(tag="Critic")

    if not os.path.exists("./wgan_image"):
        os.mkdir("./wgan_image")

    G_trainer = C.Trainer(G_fake, (G_loss, None), [G_learner], [G_progress_printer])
    C_trainer = C.Trainer(C_real, (C_loss, None), [C_learner], [C_progress_printer])

    input_map = {x: train_reader.streams.image}

    #
    # train Wasserstein GAN
    #
    logging = {"step": [], "G_loss": [], "C_loss": [], "GP": []}
    for step in range(iteration):
        #
        # train critic
        #
        C_step_loss = 0
        for _ in range(n_critic):
            z_data = np.ascontiguousarray(np.random.normal(size=(minibatch_size, z_dim)), dtype="float32")
            x_data = train_reader.next_minibatch(minibatch_size, input_map=input_map)

            batch_input = {x: x_data[x].data, z: z_data}

            #
            # compute gradient penalty
            #
            gradient = C_real.grad({C_real.arguments[0]: interpolate.eval(batch_input)})
            gp_norm = np.square(C.reduce_l2(gradient, axis=(1, 2, 3)).eval() - 1)
            batch_input[gradient_penalty] = gp_norm

            C_trainer.train_minibatch(batch_input)
            C_step_loss += C_trainer.previous_minibatch_loss_average
        C_step_loss /= n_critic

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
            
            if not os.path.exists("./wgan_image/step%d" % step):
                os.mkdir("./wgan_image/step%d" % step)

            cv2.imwrite("./wgan_image/step%d/fake.png" % step, image)

        #
        # loss and error logging
        #
        logging["step"].append(step + 1)
        logging["G_loss"].append(G_step_loss)
        logging["C_loss"].append(C_step_loss)
        logging["GP"].append(gp_norm.mean())

        G_trainer.summarize_training_progress()
        C_trainer.summarize_training_progress()

    #
    # save model and logging
    #
    G_fake.save("./wgan_generator.model")
    C_real.save("./wgan_critic.model")
    print("Saved model.")

    df = DataFrame(logging)
    df.to_csv("./wgan.csv", index=False)
    print("Saved logging.")
    
