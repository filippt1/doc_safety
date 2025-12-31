import tensorflow as tf
from tensorflow.keras import layers, Model, Input


# we import the three models to be able to infer with them
class DnCNN(Model):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=17):
        super(DnCNN, self).__init__()
        conv_args = {'padding': 'same', 'use_bias': True, 'kernel_initializer': 'orthogonal'}

        self.head_conv = layers.Conv2D(nc, 3, **conv_args)
        self.head_act = layers.ReLU()

        self.body_layers = []
        for _ in range(nb - 2):
            self.body_layers.append([
                layers.Conv2D(nc, 3, **conv_args),
                layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
                layers.ReLU()
            ])
        self.tail_conv = layers.Conv2D(out_nc, 3, **conv_args)

    def call(self, x, training=False):
        out = self.head_conv(x)
        out = self.head_act(out)
        for conv, bn, act in self.body_layers:
            out = conv(out)
            out = bn(out, training=training)
            out = act(out)
        noise = self.tail_conv(out)
        return x - noise


class ResBlock(layers.Layer):
    def __init__(self, nc):
        super(ResBlock, self).__init__()
        self.conv1 = layers.Conv2D(nc, 3, padding='same', use_bias=True, kernel_initializer='orthogonal')
        self.act = layers.ReLU()
        self.conv2 = layers.Conv2D(nc, 3, padding='same', use_bias=True, kernel_initializer='orthogonal')

    def call(self, x):
        res = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + res


class DRUNet(Model):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4):
        super(DRUNet, self).__init__()
        args = {'padding': 'same', 'use_bias': True, 'kernel_initializer': 'orthogonal'}
        self.head = layers.Conv2D(nc[0], 3, padding='same', use_bias=True)

        self.down0_blocks = [ResBlock(nc[0]) for _ in range(nb)]
        self.down0_trans = layers.Conv2D(nc[1], 2, strides=2, **args)

        self.down1_blocks = [ResBlock(nc[1]) for _ in range(nb)]
        self.down1_trans = layers.Conv2D(nc[2], 2, strides=2, **args)

        self.down2_blocks = [ResBlock(nc[2]) for _ in range(nb)]
        self.down2_trans = layers.Conv2D(nc[3], 2, strides=2, **args)

        self.down3_blocks = [ResBlock(nc[3]) for _ in range(nb)]

        self.up2_trans = layers.Conv2DTranspose(nc[2], 2, strides=2, **args)
        self.up2_fusion = layers.Conv2D(nc[2], 3, **args)
        self.up2_blocks = [ResBlock(nc[2]) for _ in range(nb)]

        self.up1_trans = layers.Conv2DTranspose(nc[1], 2, strides=2, **args)
        self.up1_fusion = layers.Conv2D(nc[1], 3, **args)
        self.up1_blocks = [ResBlock(nc[1]) for _ in range(nb)]

        self.up0_trans = layers.Conv2DTranspose(nc[0], 2, strides=2, **args)
        self.up0_fusion = layers.Conv2D(nc[0], 3, **args)
        self.up0_blocks = [ResBlock(nc[0]) for _ in range(nb)]

        self.tail = layers.Conv2D(out_nc, 3, **args)

    def call(self, x, training=False):
        x0 = self.head(x)

        # Encoder
        for block in self.down0_blocks: x0 = block(x0)
        x1 = self.down0_trans(x0)

        for block in self.down1_blocks: x1 = block(x1)
        x2 = self.down1_trans(x1)

        for block in self.down2_blocks: x2 = block(x2)
        x3 = self.down2_trans(x2)

        for block in self.down3_blocks: x3 = block(x3)

        d2 = self.up2_trans(x3)
        d2 = tf.concat([d2, x2], axis=-1)
        d2 = self.up2_fusion(d2)
        for block in self.up2_blocks: d2 = block(d2)

        d1 = self.up1_trans(d2)
        d1 = tf.concat([d1, x1], axis=-1)
        d1 = self.up1_fusion(d1)
        for block in self.up1_blocks: d1 = block(d1)

        d0 = self.up0_trans(d1)
        d0 = tf.concat([d0, x0], axis=-1)
        d0 = self.up0_fusion(d0)
        for block in self.up0_blocks: d0 = block(d0)

        return self.tail(d0)


class Pix2Pix(Model):
    def __init__(self, in_nc=3, out_nc=3):
        super(Pix2Pix, self).__init__()
        self.discriminator = self.build_discriminator(in_nc, out_nc)
        self.generator = self.build_generator(in_nc, out_nc)

    def compile(self, g_optimizer, d_optimizer, gen_loss_fn, disc_loss_fn):
        super(Pix2Pix, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm: result.add(layers.BatchNormalization())
        result.add(layers.LeakyReLU())
        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                          kernel_initializer=initializer, use_bias=False))
        result.add(layers.BatchNormalization())
        if apply_dropout: result.add(layers.Dropout(0.5))
        result.add(layers.ReLU())
        return result

    def build_generator(self, in_nc, out_nc):
        inputs = Input(shape=[256, 256, in_nc])

        down_stack = [
            self.downsample(64, 4, False), self.downsample(128, 4), self.downsample(256, 4),
            self.downsample(512, 4), self.downsample(512, 4), self.downsample(512, 4),
            self.downsample(512, 4), self.downsample(512, 4)
        ]
        up_stack = [
            self.upsample(512, 4, True), self.upsample(512, 4, True), self.upsample(512, 4, True),
            self.upsample(512, 4), self.upsample(256, 4), self.upsample(128, 4), self.upsample(64, 4)
        ]

        last = layers.Conv2DTranspose(out_nc, 4, strides=2, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                      activation='tanh')
        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])

        return Model(inputs=inputs, outputs=last(x))

    def build_discriminator(self, in_nc, out_nc):
        inp, tar = Input(shape=[256, 256, in_nc]), Input(shape=[256, 256, out_nc])
        x = layers.Concatenate()([inp, tar])

        down1 = self.downsample(64, 4, False)(x)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)

        zero_pad1 = layers.ZeroPadding2D()(down3)
        conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02),
                             use_bias=False)(zero_pad1)
        leaky = layers.LeakyReLU()(layers.BatchNormalization()(conv))

        last = layers.Conv2D(1, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02))(
            layers.ZeroPadding2D()(leaky))
        return Model(inputs=[inp, tar], outputs=last)

    def train_step(self, data):
        input_image, target = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real = self.discriminator([input_image, target], training=True)
            disc_gen = self.discriminator([input_image, gen_output], training=True)

            gan_loss = self.gen_loss_fn(disc_gen, gen_output, target)
            disc_loss = self.disc_loss_fn(disc_real, disc_gen)

        grads_g = gen_tape.gradient(gan_loss, self.generator.trainable_variables)
        grads_d = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))
        return {"gen_loss": gan_loss, "disc_loss": disc_loss}

    def test_step(self, data):
        input_image, target = data

        gen_output = self.generator(input_image, training=False)
        disc_real = self.discriminator([input_image, target], training=False)
        disc_gen = self.discriminator([input_image, gen_output], training=False)

        gan_loss = self.gen_loss_fn(disc_gen, gen_output, target)
        disc_loss = self.disc_loss_fn(disc_real, disc_gen)

        return {"gen_loss": gan_loss, "disc_loss": disc_loss}

    def call(self, x):
        return self.generator(x)
