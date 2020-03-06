'''
Class for quadratic-norm on subsampled 2D Fourier measurements
Jianxing Liao, CIG, WUSTL, 2018
Based on MATLAB code by U. S. Kamilov, CIG, WUSTL, 2017
'''
from DataFidelities.DataClass import DataClass
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import decimal



class MRIClass(DataClass):
    def __init__(self, y, mask):
        self.y = y
        self.mask = mask
        self.sigSize = mask.shape
        
    def size(self):
        sigSize = self.sigSize
        return sigSize

    def evl(self,x):
        z = self.fmult(x, self.mask)
        d = 0.5 * np.power(np.linalg.norm(self.y.flatten('F')-z.flatten('F')),2)
        return d
    
    def grad(self,x):
        z = self.fmult(x, self.mask)
        g = self.ftran(z - self.y, self.mask)
        g = g.real
        d = 0.5 * np.power(np.linalg.norm(self.y.flatten('F')-z.flatten('F')),2)
        return g,d
    

    
    @staticmethod
    def genMask(imgSize, numLines):
        if imgSize[0] % 2 != 0 or imgSize[1] % 2 != 0:
            sys.stderr.write('image must be even sized! ')
            sys.exit(1)
        center = np.ceil(imgSize/2)+1
        freqMax = math.ceil(np.sqrt(np.sum(np.power((imgSize/2),2),axis=0)))
        ang = np.linspace(0, math.pi, num=numLines+1)
        mask = np.zeros(imgSize, dtype=bool)
        
        for indLine in range(0,numLines):
            ix = np.zeros(2*freqMax + 1)
            iy = np.zeros(2*freqMax + 1)
            ind = np.zeros(2*freqMax + 1, dtype=bool)
            for i in range(2*freqMax + 1):
                ix[i] = decimal.Decimal(center[1] + (-freqMax+i)*math.cos(ang[indLine])).quantize(0,rounding=decimal.ROUND_HALF_UP)
            for i in range(2*freqMax + 1):
                iy[i] = decimal.Decimal(center[0] + (-freqMax+i)*math.sin(ang[indLine])).quantize(0,rounding=decimal.ROUND_HALF_UP)
                 
            for k in range(2*freqMax + 1):
                if (ix[k] >= 1) & (ix[k] <= imgSize[1]) & (iy[k] >= 1) & (iy[k] <= imgSize[0]):
                    ind[k] = True
                else:
                    ind[k] = False
                
            ix = ix[ind]
            iy = iy[ind]
            ix = ix.astype(np.int64)
            iy = iy.astype(np.int64)
            
            for i in range(len(ix)):
                mask[iy[i]-1][ix[i]-1] = True
        
        return mask
     
    @staticmethod
    def fmult(x,mask):
        numPix = np.shape(mask)[0] * np.shape(mask)[1]
        z = np.multiply(mask,np.fft.fftshift(np.fft.fft2(x))) / math.sqrt(numPix)
        return z
    
    @staticmethod
    def ftran(z,mask):
        numPix = len(mask)*len(mask[0])
        x = np.multiply(np.fft.ifft2(np.fft.ifftshift(np.multiply(mask, z))), np.sqrt(numPix))
        return x
