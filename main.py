from DataFidelities.MRIClass import MRIClass
from Regularizers.robjects import *
from iterAlgs import *
from util import *
import settings

import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import os

####################################################
####              HYPER-PARAMETERS               ###
####################################################
opt = settings.opt

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

for arg in vars(opt):
    print(arg, ':', getattr(opt, arg))

####################################################
####              DATA PREPARATION               ###
####################################################

# prepare workspace
np.random.seed(0)

# function for evaluating SNR
evaluateSNR = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))

# load image
data_mat = spio.loadmat('data/{}'.format(opt.data_name), squeeze_me=True)
x = data_mat['img']

# normalize x to 0-1.0
x = (x-x.min())/(x.max()-x.min())

imgSize = np.array(x.shape)

# measure
numLines = 36 # 60 # 48
mask = MRIClass.genMask(imgSize, numLines)
masksum=mask.sum()
y = MRIClass.fmult(x,mask)

# save the IFFT result
x_fft = cal_fft(x)
subsample_x_fft = x_fft * mask
ifft_subsampled_x = np.array(cal_ifft(subsample_x_fft))
plt.figure(2)
plt.imsave('IFFT_36lines_MRI_Knee_{}.jpg'.format(opt.data_name), ifft_subsampled_x, cmap='gray')

# prepare ground truth xtrue
xtrue = x

####################################################
####            NETWORK INITIALIZATION           ###
####################################################
MultiHeadRNNClass = MultiHeadRNN(x.shape, sigma=opt.sigma)

####################################################
####                    PnP                      ###
####################################################
mriObj = MRIClass(y, mask)
rObj = MultiHeadRNNClass
# optimize with APGM
recon, out = apgmEst(mriObj, rObj, numIter=opt.num_iter, step=1., accelerate=False, stochastic=False, mini_batch=20, verbose=True,
                     is_save=True, save_path='line36_apgm_result_{}_sigma{}'.format(opt.data_name, opt.sigma), xtrue=xtrue)
