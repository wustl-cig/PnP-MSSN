# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Modified on Feb, 2018 based on the work of jakeret

author: yusun
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import scipy.io as sio
import scipy.misc as smisc
from scipy.optimize import fminbound


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def to_double(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    return img

def save_mat(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    
    sio.savemat(path, {'img':img})


def save_img(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    img = to_rgb(img)
    smisc.imsave(path, img.round().astype(np.uint8))



def addwagon(x,inputSnr):
    noiseNorm = np.linalg.norm(x.flatten('F')) * 10**(-inputSnr/20)
    xBool = np.isreal(x)
    real = True
    for e in np.nditer(xBool):
        if e == False:
            real = False
    if (real == True):
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1])
    else:
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1]) + 1j * np.random.randn(np.shape(x)[0],np.shape(x)[1])
    
    noise = noise/np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = x + noise
    return y, noise

def cal_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def cal_ifft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def extract_distinct_patches(x, num_blocks, block_size):
    patches = np.zeros([num_blocks, block_size, block_size])
    nx,ny = x.shape
    count = 0
    for i in range(0,nx-block_size+1,block_size):
        for j in range(0,ny-block_size+1,block_size):
            patches[count,:] = x[i:i+block_size, j:j+block_size]
            count = count+1
    return patches

def putback_distinct_patches(patches):
    num_blocks,block_size,_ = patches.shape
    nx = ny = int(np.sqrt(num_blocks)*block_size)
    x = np.zeros([nx,ny])
    count = 0
    for i in range(0,nx-block_size+1,block_size):
        for j in range(0,ny-block_size+1,block_size):
            x[i:i+block_size, j:j+block_size] = patches[count]
            count = count+1
    return x

def addwgn(x, inputSnr):
    noiseNorm = np.linalg.norm(x.flatten('F')) * 10**(-inputSnr/20)
    xBool = np.isreal(x)
    real = True
    for e in np.nditer(xBool):
        if e == False:
            real = False
    if (real == True):
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1])
    else:
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1]) + 1j * np.random.randn(np.shape(x)[0],np.shape(x)[1])
    
    noise = noise/np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = x + noise
    return y, noise

def optimizeTau(x, algoHandle, taurange, maxfun=20):
    # maxfun ~ number of iterations for optimization

    evaluateSNR = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))
    fun = lambda tau: -evaluateSNR(x,algoHandle(tau)[0])
    tau = fminbound(fun, taurange[0],taurange[1], xtol = 1e-6, maxfun = maxfun, disp = 3)
    return tau

def powerIter(A, imgSize, iter=100, tol=1e-6, verbose=False):
    # compute singular value for A'*A
    # A should be a function (lambda:x)

    x = np.random.randn(imgSize[0],imgSize[1])
    x = x / np.linalg.norm(x.flatten('F'))

    lam = 1

    for i in range(iter):
        # apply Ax
        xnext = A(x)
        
        # xnext' * x / norm(x)^2
        lamNext = np.dot(xnext.flatten('F'), x.flatten('F')) / np.linalg.norm(x.flatten('F'))**2
        # only take the real part
        lamNext = lamNext.real   

        # normalize xnext 
        xnext = xnext / np.linalg.norm(xnext.flatten('F'))

        # compute relative difference
        relDiff = np.abs(lamNext-lam) / np.abs(lam)
        
        # verbose
        if verbose:
            print('[{}/{}] lam = {}, relative Diff = {:0.4f}'.format(i, iter, lam, relDiff))

        # stopping criterion
        if relDiff < tol:
            break

        lam = lamNext

    return lam