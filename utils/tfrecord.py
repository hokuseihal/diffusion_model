import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import torch


class TFRDataloader():
    def _parse_image_function(self, example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'index': tf.io.FixedLenFeature([], tf.int64)
        })

        image = tf.image.decode_jpeg(parsed_example['image'], channels=3)
        index = parsed_example['index']
        image = tf.cast(image, tf.float32) / 255.0
        if self.size is not None:
            s = tf.shape(image)
            minsize = tf.minimum(s[0], s[1])
            image = tf.image.resize_with_crop_or_pad(image, minsize, minsize)
            image = tf.image.resize(image, [self.size, self.size])
        image = tf.transpose(image, [2, 0, 1])
        return image, index

    def __init__(self, path, batch, s, m, size=None,get_index=False):
        self.path = path
        self.size = size
        self.batch = batch
        self.m = m
        self.s = s
        self.get_index=get_index
        self.setdataset()

    def setdataset(self):
        self.tfdataset = tf.data.TFRecordDataset(self.path) \
            .map(self._parse_image_function) \
            .prefetch(5) \
            .batch(self.batch) \
            .as_numpy_iterator()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            img, index = next(self.tfdataset)
            if self.get_index:
                return torch.from_numpy(img - self.m) / self.s, index
            else:
                return torch.from_numpy(img - self.m) / self.s
        except StopIteration:
            self.setdataset()
            raise StopIteration

    def __len__(self):
        return 202589 // self.batch


def maketfrecord(txtpath, tfrpath):
    def _bytes_feature(value):

        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(image_string, index):
        feature = {
            'image': _bytes_feature(image_string),
            'index': _int64_feature(index)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    with open(txtpath) as f:
        pathes = f.readlines()
    with tf.io.TFRecordWriter(tfrpath) as writer:
        for path in pathes:
            path = path.strip()
            print(path)
            index = int(path.split('/')[-1].split('.')[0])
            with open(path, 'rb') as img:
                img_str = img.read()
            writer.write(serialize_example(img_str, index))


if __name__ == '__main__':
    # tfdataset=tf.data.TFRecordDataset(path).map(_parse_image_function).as_numpy_iterator()
    maketfrecord('train.txt', 'tmp.tfrecord')
    # path = 'tmp.tfrecord'
    # for i, x in enumerate(TFRDataloader(path=path, size=128, epoch=1, batch=1,s=1,m=0)):
    #     print(x.permute(0,3,2,1).reshape(-1,3).mean(0))
    #
    # loader=TFRDataloader('../data/celeba.tfrecord',1,2048,0,0)
    loader = TFRDataloader('tmp.tfrecord', batch=64, s=1, m=0, size=128)
    for idx, x in enumerate(loader):
        print(x)
