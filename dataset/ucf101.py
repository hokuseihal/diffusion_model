import glob

from utils import dataset as U


class UCF101Dataset:
    def __init__(self, repath, numframe, stride, size):
        self.path = glob.glob(repath)
        self.numframe = numframe
        self.stride = stride
        self.size = size

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        return U.readvideo(path=self.path[idx], stride=self.stride, numframe=self.numframe, size=self.size)


if __name__ == '__main__':
    ds = UCF101Dataset('../data/ucf101/*.avi', 100,1,128)
    U.writevideo(ds[0],'tmp.mp4',25)
