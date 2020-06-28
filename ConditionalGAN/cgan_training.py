import cntk as C
import cv2
import numpy as np
import os
import pandas as pd

from cntk.layers import BatchNormalization, Convolution2D, ConvolutionTranspose2D, Dense

z_dim = 100
input_dim = 784
label_dim = 10

iteration = 10000
k = 2
minibatch_size = 128


def create_reader(map_file, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(map_file, C.io.StreamDefs(
        feature=C.io.StreamDef(field="features", shape=input_dim, is_sparse=False),
        label=C.io.StreamDef(field="labels", shape=label_dim, is_sparse=False))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def cgan_generator(z, y):
    with C.layers.default_options(init=C.normal(scale=0.02), bias=False, map_rank=1, use_cntk_engine=True):
        h = C.splice(z, y, axis=0)

        h = C.relu(BatchNormalization()(Dense(1024)(h)))
        h = C.relu(BatchNormalization()(Dense((128, 7, 7))(h)))
        h = C.relu(BatchNormalization()(ConvolutionTranspose2D(
            (5, 5), 128, strides=(2, 2), pad=True, output_shape=(14, 14))(h)))
        h = ConvolutionTranspose2D((5, 5), 1, activation=C.sigmoid, strides=(2, 2), pad=True, output_shape=(28, 28))(h)

    return C.reshape(h, input_dim)


def cgan_discriminator(x, y):
    with C.layers.default_options(init=C.normal(scale=0.02), map_rank=1, use_cntk_engine=True):
        hx = C.reshape(x, (1, 28, 28))
        hy = C.ones_like(hx) * C.reshape(y, (label_dim, 1, 1))
        h = C.splice(hx, hy, axis=0)

        h = C.leaky_relu((Convolution2D((5, 5), 1, strides=(2, 2))(h)), alpha=0.2)
        h = C.leaky_relu(BatchNormalization()(Convolution2D((5, 5), 64, strides=(2, 2))(h)), alpha=0.2)
        h = C.leaky_relu(BatchNormalization()(Dense(1024)(h)), alpha=0.2)

        h = Dense(1, activation=C.sigmoid)(h)

    return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./Train-28x28_cntk_text.txt", is_train=True)

    #
    # latent, input, generator, and discriminator
    #
    z = C.input_variable(shape=(z_dim,), dtype="float32")
    y = C.input_variable(shape=(label_dim,), dtype="float32")
    x = C.input_variable(shape=(input_dim,), dtype="float32")
    x_real = x / 255.0

    G_fake = cgan_generator(z, y)
    D_real = cgan_discriminator(x_real, y)
    D_fake = D_real.clone(method="share", substitutions={x_real.output: G_fake.output, y: y})

    #
    # loss function
    #
    G_loss = - C.log(D_fake)
    D_loss = - (C.log(D_real) + C.log(1.0 - D_fake))

    #
    # optimizer
    #
    G_learner = C.adam(G_fake.parameters, lr=C.learning_parameter_schedule_per_sample(2e-4), momentum=0.5,
                       gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    D_learner = C.adam(D_real.parameters, lr=C.learning_parameter_schedule_per_sample(2e-4), momentum=0.5,
                       gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)

    G_progress_printer = C.logging.ProgressPrinter(tag="Generator")
    D_progress_printer = C.logging.ProgressPrinter(tag="Discriminator")

    if not os.path.exists("./cgan_image"):
        os.mkdir("./cgan_image")

    G_trainer = C.Trainer(G_fake, (G_loss, None), [G_learner], [G_progress_printer])
    D_trainer = C.Trainer(D_real, (D_loss, None), [D_learner], [D_progress_printer])

    input_map = {x: train_reader.streams.feature, y: train_reader.streams.label}

    #
    # training
    #
    logging = {"step": [], "G_loss": [], "D_loss": []}
    for step in range(iteration):
        #
        # discriminator
        #
        for _ in range(k):
            z_data = np.random.uniform(low=-1.0, high=1.0, size=(minibatch_size, z_dim)).astype("float32")
            xy_data = train_reader.next_minibatch(minibatch_size, input_map=input_map)

            batch_input = {x: xy_data[x].data, y: xy_data[y].data, z: z_data}

            D_trainer.train_minibatch(batch_input)
            D_step_loss = D_trainer.previous_minibatch_loss_average

        #
        # generator
        #
        z_data = np.random.uniform(low=-1.0, high=1.0, size=(minibatch_size, z_dim)).astype("float32")
        xy_data = train_reader.next_minibatch(minibatch_size, input_map=input_map)

        batch_input = {z: z_data, y: xy_data[y].data}

        output = G_trainer.train_minibatch(batch_input, outputs=[G_fake])
        G_step_loss = G_trainer.previous_minibatch_loss_average

        #
        # tensorboard image
        #
        if step % 1000 == 0:
            image = np.reshape(list(output[1].values())[0][0], (28, 28)) * 255
            
            if not os.path.exists("./cgan_image/step%d" % step):
                os.mkdir("./cgan_image/step%d" % step)

            cv2.imwrite("./cgan_image/step%d/fake.png" % step, image)

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
    G_fake.save("./cgan_generator.model")
    D_real.save("./cgan_discriminator.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./cgan.csv", index=False)
    print("Saved logging.")
    
