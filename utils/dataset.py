import cv2
import imageio
import torch


def readvideo(path, size, stride=1, numframe=100):
    buf = torch.zeros(numframe, size, size, 3, dtype=torch.uint8)
    reader = imageio.get_reader(path)
    t = 0
    for idx, im in enumerate(reader):
        if idx % stride == 0:
            buf[t] = torch.from_numpy(cv2.resize(im, (size, size)))
            t += 1
            if t >= numframe:
                break
    return buf.permute(3, 0, 1, 2)/255


def writevideo(x, path, fps):
    C, T, H, W = x.shape
    assert C * T * H * W != 0
    x = x.permute(1, 2, 3, 0).numpy()  # T,H,W,C
    x=(x*255).clip(0,255).astype('uint8')
    writer = imageio.get_writer(path, format='FFMPEG', mode='I', fps=fps)
    for t in range(T):
        writer.append_data(x[t])
    writer.close()


if __name__ == '__main__':
    video = readvideo('/home/hokusei/src/data/ucf101/v_Haircut_g01_c03.avi', 128, 1, numframe=100)
    print(video.shape)
    writevideo(video, 'tmp.mp4', 25)
