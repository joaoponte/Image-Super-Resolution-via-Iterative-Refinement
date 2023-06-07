import os
import math
from math import exp
import numpy as np
import cv2
from skimage.io import imsave

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from skimage.metrics import peak_signal_noise_ratio

from torchvision.utils import make_grid
import torch

import matplotlib.pyplot as plt


def tensor2img(tensor, out_type=np.uint16, min_max=(-1, 1), n_images_per_row=None, is_rgb=False, debug=False):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W) or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]

    if debug:
        print(tensor.min(), tensor.max(), tensor.dtype, tensor.shape)

    n_dim = tensor.dim()
    if n_dim == 4 and is_rgb:
        n_img = n_images_per_row if n_images_per_row else int(math.sqrt(len(tensor)))
        img_np = make_grid(tensor, nrow=n_img, normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3 and not is_rgb:
        n_img = n_images_per_row if n_images_per_row else int(math.sqrt(len(tensor)))
        img_np = make_grid(torch.reshape(tensor, (tensor.shape[0], 1, *tensor.shape[1:])), nrow=n_img, normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3 and is_rgb:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if debug:
        print(img_np.min(), img_np.max(), img_np.dtype, img_np.shape)

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    elif out_type == np.uint16:
        img_np = (img_np * np.iinfo(out_type).max).round()

    if debug:
        print(img_np.min(), img_np.max(), img_np.dtype)

    return img_np.astype(out_type)


def tensor2img_3d(tensor, out_type=np.uint16, max_value=255*255, min_max=(-1, 1), slices=None, debug=False):
    if debug:
        print(f'metrics.py > tensor2img_3d -> {tensor.shape=}, {tensor.min()=}, {tensor.max()=}, {tensor.dtype=}')
    b,c,d,h,w = tensor.shape

    if debug:
        _, axs = plt.subplots(4, 3, figsize=(12, 8), num=0)
        axs[0,0].imshow((tensor[0,0,32,:,:]+1)/2); axs[0,1].imshow((tensor[0,0,:, 32, :]+1)/2); axs[0,2].imshow((tensor[0,0,:,:,32]+1)/2)

    tensor = tensor.squeeze().float().cpu().clamp_(*min_max) # .clamp_(*min_max) # clamp -> deu problema! Porém, aparentemente é importante...
    
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    if debug:
        print(f'metrics.py > tensor2img_3d -> {tensor.shape=}, {tensor.min()=}, {tensor.max()=}, {tensor.dtype=}')
        axs[1,0].imshow(tensor[32,:,:]); axs[1,1].imshow(tensor[:, 32, :]); axs[1,2].imshow(tensor[:,:,32])
        

    img = np.zeros((3*h+4, b*w+(b-1)*2))

    divisor_h = np.zeros((2, img.shape[1]))
    divisor_v = np.zeros((img.shape[0], 2))
    if debug:
        print(f'metrics.py > tensor2img_3d -> {img.shape=}, {img.shape=}, {img.shape=}')

    if slices:
        for n in range(b):
            _img = np.concatenate((tensor[slices[0], :, :], divisor_h, tensor[:, slices[1], :], divisor_h, tensor[:, :, slices[2]]), axis=0)
            img[
                :_img.shape[0],
                n*(2+_img.shape[1]):n*2 + (n+1)*_img.shape[1],
            ] = _img
    else:
        for n in range(b):
            _img = np.concatenate((tensor[int(d/2), :, :], divisor_h, tensor[:, int(h/2), :], divisor_h, tensor[:, :, int(w/2)]), axis=0)
            if debug:
                print(f'metrics.py > tensor2img_3d > loop -> {_img.shape=}, {n*(2+_img.shape[1])=}, {n*2 + (n+1)*_img.shape[1]=}')
            img[
                :_img.shape[0],
                n*(2+_img.shape[1]):n*2 + (n+1)*_img.shape[1],
            ] = _img

            if debug:
                axs[2,0].imshow(_img[:64,:]); axs[2,1].imshow(_img[64+2:64+64+2,:]); axs[2,2].imshow(_img[64+2+64+2:,:])

    res = (max_value*img).astype(out_type) # round(0).
    if debug:
        print(f'metrics.py > tensor2img_3d -> {res.shape=}, {res.min()=}, {res.max()=}, {res.dtype=}')
        axs[3,0].imshow(res[:64,:]); axs[3,1].imshow(res[64+2:64+64+2,:]); axs[3,2].imshow(res[64+2+64+2:,:])
        plt.show()
    
    return res


def save_img(img, img_path, mode='RGB'):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)
    # print(f'Saving {img_path}', img.min(), img.max(), img.dtype, img.shape)
    imsave(img_path, img, check_contrast=False)

def save_img_3d(img, img_path):
    # print(f'metrics.py > save_img_3d -> {img_path.split("/")[-1]}, {img.shape=}, {img.min()=}, {img.max()=}, {img.mean()=}, {img.std()=}')
    # ax[0].imshow(img, interpolation='none')
    # ax[0].set_title(img_path)
    
    # ax[1].hist(img.ravel(), bins=255)
    
    imsave(img_path, img, check_contrast=False)



def calculate_psnr(img1, img2, max_value=255.):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_value / math.sqrt(mse))


def ssim(img1, img2, max_value=255.):
    C1 = (0.01 * max_value)**2
    C2 = (0.03 * max_value)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
# Search functions that calculate PSNR, SSIM, FID, LPIPS, ... for 3D data.
def psnr_3d(img1, img2, max_value=255.):
    return peak_signal_noise_ratio(img1.numpy(), img2.numpy(), data_range=max_value)


# Copied from the repo:
# https://github.com/jinh0park/pytorch-ssim-3D/blob/master/pytorch_ssim/__init__.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def ssim_3d(img1, img2, max_value=255.):
    # return ssim3D(img1, img2)
    pass

def fid_3d(img1, img2, max_value=255.):
    pass

def lpips_3d(img1, img2, max_value=255.):
    pass