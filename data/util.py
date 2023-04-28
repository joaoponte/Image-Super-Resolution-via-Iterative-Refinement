import os
import torch
import torchvision
import random
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / img.max()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def transform2PIL(img, rgb=False, value_range=None, min_max=(0, 1)):
    """
    from numpy array to PIL
    """

    # if img.dtype == np.uint8 and len(img.shape) == 2:
    #     mode = 'L'
    # elif img.dtype == np.uint8 and len(img.shape) == 3:
    #     mode = 'RGB'
    # elif img.dtype == np.uint16:
    #     mode = 'I;16'
    # elif img.dtype == np.int32:
    #     mode = 'I'
    # elif img.dtype == np.float32:
    #     mode = 'F'

    if value_range:
        img = 255*((img-value_range[0])/(value_range[1] - value_range[0]))
    else:
        img = 255*((img-img.min())/(img.max() - img.min()))
    img = img.astype(np.uint8)

    if rgb:
        img = Image.fromarray(img, mode='L').convert('RGB')
    else:
        img = Image.fromarray(img, mode='L')

    return img

def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    # img = torch.from_numpy(np.ascontiguousarray(
    #     np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
# hflip = torchvision.transforms.RandomHorizontalFlip()
hflip = torchvision.transforms.functional.hflip
def transform_augment_bkp(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

def transform_augment(img_list, img_value_ranges, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    if split == 'train' and random.random() < 0.5:
        imgs = [hflip(img) for img in imgs]

    # ret_img = [((img-vr[0])/(vr[1]-vr[0])) * (min_max[1] - min_max[0]) + min_max[0] for img, vr in zip(imgs, img_value_ranges)]
    ret_img = [(img) * (min_max[1] - min_max[0]) + min_max[0] for img, vr in zip(imgs, img_value_ranges)]
    return ret_img
