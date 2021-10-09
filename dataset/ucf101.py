import glob
import lzma
import os
import pickle as pkl

import cv2
import numpy as np
import torch
import torch.nn.functional as F

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from utils.motion_estimate import motion_estimate


def readvideo(path):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    return torch.from_numpy(buf).permute(0, 3, 1, 2)


def make_ucf_tfrecord(dir, size, batchsize, kernelsize, p, device, s=0.5, m=0.5):
    def make_example(err, motion):
        return tf.train.Example(features=tf.train.Features(feature={
            'err': tf.train.Feature(bytes_list=tf.train.BytesList(value=[err])),
            'motion': tf.train.Feature(bytes_list=tf.train.BytesList(value=[motion]))
        }))

    writer = tf.io.TFRecordWriter('tmp.tfrecord')
    for idx, path in enumerate(glob.glob(f'{dir}/*.avi')):
        print(f'read {path}')
        try:
            video = (readvideo(path) / 255 - m) / s
            T, C, H, W = video.shape
            if H < W:
                video = video[:, :, :, (W - H) // 2:-(W - H) // 2]
            else:
                video = video[:, :, (W - H) // 2:-(W - H) // 2]
            video = video.to(device)
            video = F.interpolate(video, size=(size, size))
            errs = []
            motions = []
            for i in range(0, T - 1, batchsize):
                print(f'\r process frame:{i}/{T}', end='')
                err, motion = motion_estimate(video[:-1][i:i + batchsize],
                                              video[i + 1:i + batchsize + 1], kernelsize, 'full', p, 1)
                errs.append(err.cpu())
                motions.append(motion.cpu())
            errs = torch.cat(errs)
            motions = torch.cat(motions)
            with open(f'../data/ucf101/pkl/{idx}.pkl', 'wb') as f:
                pkl.dump([
                    lzma.compress(pkl.dumps(errs)),
                    lzma.compress(pkl.dumps(motions)),
                    lzma.compress(pkl.dumps(video))
                ], f)
            # writer.write(make_example(zlib.compress(pkl.dumps(errs.numpy())), zlib.compress(pkl.dumps(motions.numpy()))).SerializeToString())
            print()
        except:
            print(f"ERROR:{path}")
    writer.close()


class UCF101Dataset:
    def __init__(self, pklpath, numframe):
        self.path = glob.glob(f'{pklpath}/*.pkl')
        assert len(self.path) != 0, "YOU MUST SET WRONG PATH?"
        self.numframe = numframe

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        with open(self.path[idx], 'rb') as f:
            err, motion, video = pkl.load(f)
        err = pkl.loads(lzma.decompress(err))
        motion = pkl.loads(lzma.decompress(motion))
        video = pkl.loads(lzma.decompress(video))
        start = torch.randint(video.shape[0] - self.numframe, (1,))
        retimg = video[start]
        reterr = err[start:start + self.numframe]
        retmotion = motion[start:start + self.numframe]
        return retimg, reterr, retmotion


if __name__ == '__main__':
    # make_ucf_tfrecord(dir='../data/ucf101', size=64, batchsize=48, p=8, kernelsize=8, device='cuda')
    d = UCF101Dataset('../data/ucf101/pkl', 16)
    for i in d:
        print(i)
    # def get(proto):
    #     desc={'err':tf.io.FixedLenFeature([],tf.string),'motion':tf.io.FixedLenFeature([],tf.string)}
    #     elem=tf.io.parse_single_example(proto,desc)
    #     return elem
    # ds=tf.data.TFRecordDataset('tmp.tfrecord').map(get).prefetch(1).batch(1).as_numpy_iterator()
    # for k in ds:
    #     print(pkl.loads(zlib.decompress(k['err'][0])))
    # with open('tmp.pkl','rb') as f:
    #     data=pkl.load(f)
    #     err=pkl.loads(zlib.decompress(data[0]))
    #     print(err.shape)
