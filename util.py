import os
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import poisson
from skimage.transform import resize

##save netrwork

def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               './{}/model_epoch{}.pth'.format(ckpt_dir, epoch))

##load network

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./{}/{}'.format(ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

##sampling

def add_sampling(img, type='random', opts=None):
    sz = img.shape

    if type == 'uniform':
        ds_y = opts[0].astype(np.int)  # sampling ratio
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(sz)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk

    elif type == 'random':
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        prob = opts[0]

        msk = (rnd > prob).astype(np.float)
        dst = img * msk

    elif type == 'gaussian':

        x0 = opts[0]
        y0 = opts[1]
        sigma_x = opts[2]
        sigma_y = opts[3]
        A = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])
        x, y = np.meshgrid(lx, ly)

        gaus = A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))

        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)
        dst = img * msk
    else:
        raise ValueError('No Types')

    return dst

##noising

def add_noise(img, type='random', opts=None):
    sz = img.shape

    if type == 'random':
        sigma = opts[0]
        noise = sigma / 255.0 * np.random.randn(sz[0], sz[1], sz[2])

        dst = img + noise

    elif type == 'poisson':
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    return dst

## bluring
def add_blur(img, type='bilinear',  opts=None):
    if type == 'nearest':
        order = 0
    elif type == 'bilinear':
        order = 1
    elif type == 'biquadratic':
        order = 2
    elif type == 'bicubic':
        order = 3
    elif type == 'biquartic':
        order = 4
    elif type == 'biquintic':
        order = 5
    else:
        raise ValueError('No Types')

    sz = img.shape

    dw = opts[0] #downsampling ration
    if len(opts) == 1: #downsampling된 영상을 원본 화질로 되돌릴지 그대로 가져갈지
        keepdim = True #설정이 안된경우
    else:
        keepdim = opts[1] #1이면 다시 upsampling 0이면 유지

    dst = resize(img, output_shape = (sz[0] // dw, sz[1] // dw, sz[2]), order = order)

    if keepdim:
        dst = resize(img, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst


