import torchvision.utils as tvu


def save_image(x, path, s=1, m=0):
    tvu.save_image(x * s + m, path)
