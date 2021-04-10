import cntk as C
import cntk.io.transforms as xforms
import cv2
import numpy as np
import os
import pandas as pd

from cntk.layers import BatchNormalization, Convolution2D, ConvolutionTranspose2D, Dropout

img_channel = 3
img_height = 256
img_width = 256
num_classes = 1

epoch_size = 100
minibatch_size = 1
num_samples = 378

lambda_1 = 10.0


def create_reader(map_file, is_train):
    transforms = [xforms.scale(width=img_width, height=img_height, channels=img_channel, interpolations="linear")]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        image=C.io.StreamDef(field="image", transforms=transforms),
        dummy=C.io.StreamDef(field="label", shape=num_classes))),
                                randomize=False, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def pix2pix_generator(h):
    with C.layers.default_options(init=C.normal(0.02), pad=True, bias=False, map_rank=1, use_cntk_engine=True):
        h_enc1 = C.leaky_relu(Convolution2D((4, 4), 64, strides=2, bias=True)(h), alpha=0.2)
        h_enc2 = C.leaky_relu(BatchNormalization()(Convolution2D((4, 4), 128, strides=2)(h_enc1)), alpha=0.2)
        h_enc3 = C.leaky_relu(BatchNormalization()(Convolution2D((4, 4), 256, strides=2)(h_enc2)), alpha=0.2)
        h_enc4 = C.leaky_relu(BatchNormalization()(Convolution2D((4, 4), 512, strides=2)(h_enc3)), alpha=0.2)
        h_enc5 = C.leaky_relu(BatchNormalization()(Convolution2D((4, 4), 512, strides=2)(h_enc4)), alpha=0.2)
        h_enc6 = C.leaky_relu(BatchNormalization()(Convolution2D((4, 4), 512, strides=2)(h_enc5)), alpha=0.2)
        h_enc7 = C.leaky_relu(BatchNormalization()(Convolution2D((4, 4), 512, strides=1)(h_enc6)), alpha=0.2)
        h_enc8 = C.leaky_relu(BatchNormalization()(Convolution2D((4, 4), 512, strides=1)(h_enc7)), alpha=0.2)

        h_dec8 = Dropout(0.5)(BatchNormalization()(ConvolutionTranspose2D(
            (4, 4), 512, strides=1, pad=True, output_shape=(img_height // 64, img_width // 64))(h_enc8)))
        h_dec8 = C.splice(h_dec8, h_enc8, axis=0)
        h_dec8 = C.relu(h_dec8)

        h_dec7 = Dropout(0.5)(BatchNormalization()(ConvolutionTranspose2D(
            (4, 4), 512, strides=1, pad=True, output_shape=(img_height // 64, img_width // 64))(h_dec8)))
        h_dec7 = C.splice(h_dec7, h_enc7, axis=0)
        h_dec7 = C.relu(h_dec7)

        h_dec6 = Dropout(0.5)(BatchNormalization()(ConvolutionTranspose2D(
            (4, 4), 512, strides=1, pad=True, output_shape=(img_height // 64, img_width // 64))(h_dec7)))
        h_dec6 = C.splice(h_dec6, h_enc6, axis=0)
        h_dec6 = C.relu(h_dec6)

        h_dec5 = Dropout(0.5)(BatchNormalization()(ConvolutionTranspose2D(
            (4, 4), 512, strides=2, pad=True, output_shape=(img_height // 32, img_width // 32))(h_dec6)))
        h_dec5 = C.splice(h_dec5, h_enc5, axis=0)
        h_dec5 = C.relu(h_dec5)

        h_dec4 = Dropout(0.5)(BatchNormalization()(ConvolutionTranspose2D(
            (4, 4), 512, strides=2, pad=True, output_shape=(img_height // 16, img_width // 16))(h_dec5)))
        h_dec4 = C.splice(h_dec4, h_enc4, axis=0)
        h_dec4 = C.relu(h_dec4)

        h_dec3 = Dropout(0.5)(BatchNormalization()(ConvolutionTranspose2D(
            (4, 4), 256, strides=2, pad=True, output_shape=(img_height // 8, img_width // 8))(h_dec4)))
        h_dec3 = C.splice(h_dec3, h_enc3, axis=0)
        h_dec3 = C.relu(h_dec3)

        h_dec2 = Dropout(0.5)(BatchNormalization()(ConvolutionTranspose2D(
            (4, 4), 128, strides=2, pad=True, output_shape=(img_height // 4, img_width // 4))(h_dec3)))
        h_dec2 = C.splice(h_dec2, h_enc2, axis=0)
        h_dec2 = C.relu(h_dec2)

        h_dec1 = Dropout(0.5)(BatchNormalization()(ConvolutionTranspose2D(
            (4, 4), 64, strides=2, pad=True, output_shape=(img_height // 2, img_width // 2))(h_dec2)))
        h_dec1 = C.splice(h_dec1, h_enc1, axis=0)
        h_dec1 = C.relu(h_dec1)

        h = ConvolutionTranspose2D((4, 4), 3, activation=C.tanh, strides=2, pad=True, bias=True,
                                   output_shape=(img_height, img_width))(h_dec1)

        return h


def pix2pix_discriminator(y, x):
    with C.layers.default_options(init=C.normal(0.02), pad=True, bias=False, map_rank=1, use_cntk_engine=True):
        x = C.leaky_relu(Convolution2D((3, 3), 32, strides=2, bias=True)(x), alpha=0.2)
        y = C.leaky_relu(Convolution2D((3, 3), 32, strides=2, bias=True)(y), alpha=0.2)

        h = C.splice(x, y, axis=0)

        h = C.leaky_relu(BatchNormalization()(Convolution2D((3, 3), 128, strides=2)(h)), alpha=0.2)
        h = C.leaky_relu(BatchNormalization()(Convolution2D((3, 3), 256, strides=2)(h)), alpha=0.2)
        h = C.leaky_relu(BatchNormalization()(Convolution2D((3, 3), 512, strides=2)(h)), alpha=0.2)

        h = Convolution2D((1, 1), 1, activation=None, bias=True)(h)

        return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    x_train_reader = create_reader("./train_pix2pix_x_map.txt", is_train=True)
    y_train_reader = create_reader("./train_pix2pix_y_map.txt", is_train=True)

    #
    # generator, and discriminator
    #
    x = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    y = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)

    x_real = (x - 127.5) / 127.5
    y_real = (y - 127.5) / 127.5

    G_fake = pix2pix_generator(x)
    D_real = pix2pix_discriminator(y_real, x_real)
    D_fake = D_real.clone(method="share", substitutions={y_real.output: G_fake.output, x_real.output: x_real.output})

    #
    # loss function
    #
    G_loss = C.reduce_mean(C.square(D_fake - 1.0)) / 2 + lambda_1 * C.reduce_mean(C.abs(y_real - G_fake))
    D_loss = C.reduce_mean(C.square(D_real - 1.0)) / 2 + C.reduce_mean(C.square(D_fake)) / 2

    #
    # optimizer and cyclical learning rate
    #
    G_learner = C.adam(G_fake.parameters, lr=1e-4, momentum=0.5,
                       gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    D_learner = C.adam(D_real.parameters, lr=1e-4, momentum=0.5,
                       gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    G_progress_printer = C.logging.ProgressPrinter(tag="Generator")
    D_progress_printer = C.logging.ProgressPrinter(tag="Discriminator")

    if not os.path.exists("./pix2pix_image"):
        os.mkdir("./pix2pix_image")

    G_trainer = C.Trainer(G_fake, (G_loss, None), [G_learner], [G_progress_printer])
    D_trainer = C.Trainer(D_real, (D_loss, None), [D_learner], [D_progress_printer])

    input_map = {x: x_train_reader.streams.image}
    truth_map = {y: y_train_reader.streams.image}

    #
    # training
    #
    logging = {"epoch": [], "G_loss": [], "D_loss": []}
    for epoch in range(epoch_size):
        sample_count = 0
        D_epoch_loss = 0
        G_epoch_loss = 0
        while sample_count < num_samples:
            #
            # discriminator
            #
            x_data = x_train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)
            y_data = y_train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=truth_map)

            batch_input = {x: x_data[x].data, y: y_data[y].data}

            D_trainer.train_minibatch(batch_input)
            D_epoch_loss += D_trainer.previous_minibatch_loss_average

            #
            # generator
            #
            output = G_trainer.train_minibatch(batch_input, outputs=[G_fake])
            G_epoch_loss += G_trainer.previous_minibatch_loss_average
            
            sample_count += minibatch_size
            
        #
        # tensorboard image
        #
        if epoch % 10 == 0:
            image = np.transpose(list(output[1].values())[0][0] / 2 + 0.5, (1, 2, 0)) * 255
            
            if not os.path.exists("./pix2pix_image/epoch%d" % epoch):
                os.mkdir("./pix2pix_image/epoch%d" % epoch)

            cv2.imwrite("./pix2pix_image/epoch%d/fake.png" % epoch, image)
            
        #
        # G loss and D loss logging
        #
        logging["epoch"].append(epoch + 1)
        logging["G_loss"].append(G_epoch_loss / (num_samples / minibatch_size))
        logging["D_loss"].append(D_epoch_loss / (num_samples / minibatch_size))

        G_trainer.summarize_training_progress()
        D_trainer.summarize_training_progress()

    #
    # save model and logging
    #
    G_fake.save("./pix2pix_generator.model")
    D_real.save("./pix2pix_discriminator.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./pix2pix.csv", index=False)
    print("Saved logging.")
    
