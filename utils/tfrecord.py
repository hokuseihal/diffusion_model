import os

import tensorflow as tf
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TFRDataloader():
    def _parse_image_function(self, example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, {'image': tf.io.FixedLenFeature([], tf.string)})

        image = tf.image.decode_jpeg(parsed_example['image'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        if self.size is not None:
            s = tf.shape(image)
            minsize = tf.minimum(s[0], s[1])
            image = tf.image.resize_with_crop_or_pad(image, minsize, minsize)
            image = tf.image.resize(image, [self.size, self.size])
        image = tf.transpose(image, [2, 0, 1])
        return image

    def __init__(self, path, epoch, batch, s, m, size=None):
        self.size = size
        self.batch = batch
        self.epoch = epoch
        self.tfdataset = tf.data.TFRecordDataset(path) \
            .map(self._parse_image_function) \
            .repeat(epoch) \
            .prefetch(5) \
            .batch(batch) \
            .as_numpy_iterator()
        self.m = m
        self.s = s


    def __iter__(self):
        return self

    def __next__(self):
        return (torch.from_numpy(next(self.tfdataset)) - self.m) / self.s

    def __len__(self):
        return 202589 // self.batch


if __name__ == '__main__':
    # tfdataset=tf.data.TFRecordDataset(path).map(_parse_image_function).as_numpy_iterator()
    path = '/home/hokusei/src/data/celeba.tfrecord'
    for i, x in enumerate(TFRDataloader(path=path, size=128, epoch=1, batch=4096,s=1,m=0)):
        print(x.permute(0,3,2,1).reshape(-1,3).mean(0))
