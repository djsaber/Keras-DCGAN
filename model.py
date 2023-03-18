# coding=gbk

from keras.layers import Dense, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose
from keras.layers import Input, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm


class Generator(Model):
    '''生成器，从随机噪声中生成样本'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            [Dense(4*4*1024), 
             Reshape((4, 4, 1024)),
             BatchNormalization(momentum=0.8),
             LeakyReLU(alpha=0.2)],
            [Conv2DTranspose(512, 3, strides=2, padding='same'), 
             BatchNormalization(momentum=0.8),
             LeakyReLU(alpha=0.2)],
            [Conv2DTranspose(256, 3, strides=2, padding='same'), 
             BatchNormalization(momentum=0.8),
             LeakyReLU(alpha=0.2)],
            [Conv2DTranspose(128, 3, strides=2, padding='same'), 
             BatchNormalization(momentum=0.8),
             LeakyReLU(alpha=0.2)],
            [Conv2DTranspose(3, 3, strides=2, padding='same', activation='tanh')]
            ]

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))


class Discriminator(Model):
    '''判别器，判别生成样本和真实样本'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            [Conv2D(64, 3, 2, padding='same'),
             LeakyReLU(alpha=0.2)],
            [Conv2D(128, 3, 2, padding='same'),
             LeakyReLU(alpha=0.2)],
            [Conv2D(256, 3, 2, padding='same'),
             LeakyReLU(alpha=0.2)],
            [Conv2D(512, 3, 2, padding='same'),
             LeakyReLU(alpha=0.2)],
            [Flatten(),
             Dense(2, activation='softmax')]
            ]

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))
 
        
class DCGAN():
    def __init__(self):
        self.output_shape = (64, 64, 3)
        self.latent_shape = (100, )
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.discriminator.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(0.0002, 0.5),
            metrics=['acc'])
        self.combined = self.combined_model()
        self.combined.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(0.0002, 0.5),
            metrics=['acc'])
        
    def combined_model(self):
        _in = Input(self.latent_shape)
        x = self.generator(_in)
        self.discriminator.trainable = False
        x = self.discriminator(x)
        return Model(inputs=_in, outputs=x)

    def fit(
        self, 
        data_generator, 
        epochs, 
        batch_size, 
        steps_per_epoch,
        ):

        # 软标签
        valid = K.one_hot(K.ones((batch_size, ), dtype='int32'), num_classes=2)
        fake = K.one_hot(K.zeros((batch_size, ), dtype='int32'), num_classes=2)
        valid = valid*0.8 + 0.1
        fake = fake*0.8 + 0.1

        for epoch in range(epochs):
            for _ in tqdm(range(steps_per_epoch), desc=f'epoch：{epoch}'):
                # 训练判别器
                for _ in range(2):
                    imgs, _ = next(data_generator)
                    if imgs.shape[0] != batch_size:
                        imgs, _ = next(data_generator)
                    gen_imgs = self.sample_imgs(batch_size)
                    d_loss_real, d_acc_real = self.discriminator.train_on_batch(imgs, valid)
                    d_loss_gen, d_acc_gen = self.discriminator.train_on_batch(gen_imgs, fake)

                # 训练生成器
                for _ in range(1):
                    noise = K.random_normal((batch_size, *self.latent_shape), mean=0, stddev=1)
                    g_loss, g_acc = self.combined.train_on_batch(noise, valid)

            print(f" - epoch：{epoch} D real [loss: {d_loss_real}, acc: {d_acc_real}]")
            print(f" - epoch：{epoch} D gen [loss: {d_loss_gen}, acc: {d_acc_gen}]")
            print(f" - epoch：{epoch} G [loss: {g_loss}, acc: {g_acc}]")

    def save_weights(self, path):
        self.generator.save_weights(path+'generator.h5')
        self.discriminator.save_weights(path+'discriminator.h5')

    def load_weights(self, path):
        self.generator.load_weights(path+'generator.h5')
        self.discriminator.load_weights(path+'discriminator.h5')

    def sample_imgs(
        self, 
        batch_size, 
        save_path=None, 
        **kwargs):
        noise = K.random_normal((batch_size, *self.latent_shape), mean=0, stddev=1)
        gen_imgs = self.generator.predict(noise, batch_size, verbose='0')
        if save_path:
            gen_imgs = gen_imgs * 0.5 + 0.5
            for i in range(batch_size):
                plt.imsave(f'{save_path}{i}.png', gen_imgs[i], **kwargs)
        return gen_imgs