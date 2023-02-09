import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    tp = patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img[:,:,:n_channels]

    return [_set_channel(a) for a in args]

def get_patch2(*args, patch_size=96, scale=1, scale2=1):
    ih, iw = args[0].shape[:2]  ## LR image

    tp = int(round((scale/2) * patch_size)) * 2
    tp2 = int(round((scale2/2) * patch_size)) * 2
    ip = patch_size

    if scale==int(scale):
        step = 1
    elif (scale*2)== int(scale*2):
        step = 2
    elif (scale*5) == int(scale*5):
        step = 5
    else:
        step = 10
    if scale2==int(scale2):
        step2 = 1
    elif (scale2*2)== int(scale2*2):
        step2 = 2
    elif (scale2*5) == int(scale2*5):
        step2 = 5
    else:
        step2 = 10

    iy = random.randrange(2, (ih-ip)//step-2) * step
    ix = random.randrange(2, (iw-ip)//step2-2) * step2

    tx, ty = int(round(scale2 * ix)), int(round(scale * iy))

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp2, :] for a in args[1:]]
    ]

    return ret

def get_patch3(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]
    
    ip = 48
    query_size=int(round(ip*scale/2))
    if query_size%3!=0:
        query_size=query_size-(query_size%3)
    tp=query_size*2
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = int(round(scale * ix)), int(round(scale * iy))
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def get_patch4(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[-2], args[0].shape[-1]
  
    ip = 48
    query_size=int(round(ip*scale/2))
    if query_size%1!=0:
        query_size=query_size-(query_size%1)
    tp=query_size*2
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = int(round(scale * ix)), int(round(scale * iy))
    else:
        tx, ty = ix, iy

    ret = [
        args[0][:, :, iy:iy + ip, ix:ix + ip],
        *[a[:, :, ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return ret

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

