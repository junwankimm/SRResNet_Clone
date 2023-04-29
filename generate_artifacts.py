import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.io import loadmat

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale
##

img = plt.imread('lena.png')

# img = np.mean(img, axis=2, keepdims=True) #Grayscale

sz = img.shape
cmap = 'gray' if sz[2] == 1 else None

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')
plt.show()

## Inpainting : Uniform Smapling

ds_y = 2 #sampling ratio
ds_x = 4

msk = np.zeros(sz)
msk[::ds_y, ::ds_x, :] = 1

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.show()
##random samplling

rnd = np.random.rand(sz[0], sz[1], sz[2])
prob = 0.5

msk = (rnd > prob).astype(np.float)
dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.show()

##Gaussian Sampling

ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])
x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sigma_x = 1
sigma_y = 1
A = 1

gaus = A * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))
gaus = np.tile(gaus[:, :, np.newaxis], (1,1,sz[2]))

rnd = np.random.rand(sz[0], sz[1], sz[2])
msk = (rnd < gaus).astype(np.float)
dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.show()


##Denoising

sigma = 60.0
noise = sigma/255.0 * np.random.randn(sz[0], sz[1], sz[2])

dst = img + noise

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.show()


##poisson noise
dst = poisson.rvs(255.0 * img) / 255.0
noise = dst -img

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.show()


##CT-Domain Poisson
N = 512
ANG = 180
VIEW = 360
THETA = np.linspace(0, ANG, VIEW, endpoint=False)

A = lambda x : radon(x, THETA, circle=False).astype(np.float32)
AT = lambda y : iradon(y, THETA, circle=False, filter=None, output_size=N).astype(np.float32)
AINV = lambda y : iradon(y, THETA, circle=False, output_size=N).astype(np.float32)

pht = shepp_logan_phantom()
pht = 0.03 * rescale(pht, scale=512/400, order = 0)

prj = A(pht)

i0 = 1e4
dst = i0 * np.exp(-prj)
dst = poisson.rvs(dst)
dst = -np.log(dst / i0)
dst[dst < 0] = 0

noise = dst -prj
rec = AINV(prj)
rec_noise = AINV(noise)
rec_dst = AINV(dst)

plt.subplot(241)
plt.imshow(pht, cmap='gray', vmin=0, vmax=0.03)
plt.title('GT')

plt.subplot(242)
plt.imshow(rec, cmap='gray', vmin=0, vmax=0.03)
plt.title('GT')

plt.subplot(243)
plt.imshow(rec_noise, cmap='gray')
plt.title('GT')

plt.subplot(244)
plt.imshow(rec_dst, cmap='gray', vmin=0, vmax=0.03)
plt.title('GT')

plt.subplot(246)
plt.imshow(prj, cmap='gray')
plt.title('GT')

plt.subplot(247)
plt.imshow(noise, cmap='gray')
plt.title('GT')

plt.subplot(248)
plt.imshow(dst, cmap='gray')
plt.title('GT')

plt.show()
##superresolution

dw = 1/5.0
order = 5
dst_dw = rescale(img, scale=(dw, dw, 1), order=order)
dst_up = rescale(dst_dw, scale=(1/dw, 1/dw, 1), order=order)


plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(132)
plt.imshow(np.squeeze(dst_dw), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.subplot(133)
plt.imshow(np.squeeze(dst_up), cmap=cmap, vmin=0, vmax=1)
plt.title('GT')

plt.show()
##

