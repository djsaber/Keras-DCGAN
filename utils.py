# coding=gbk

from keras.preprocessing.image import ImageDataGenerator


def data_gen(path, batch_size, target_size, **kwargs):
    '''创建图片生成器'''

    def prep_fn(img):
        img = img / 255.0
        img = (img - 0.5) * 2
        return img

    img_gen = ImageDataGenerator(
        preprocessing_function=prep_fn,
        horizontal_flip=True,
        )

    gen = img_gen.flow_from_directory(
        path,
        batch_size=batch_size,
        target_size=target_size,
        **kwargs)

    return gen
