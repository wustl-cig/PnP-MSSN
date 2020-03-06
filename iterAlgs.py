# library
import os
import shutil
import tensorflow as tf
import scipy.io as spio
import numpy as np
import warnings
import time
# scripts
import util
import time

######## Iterative Methods #######

def apgmEst(dObj, rObj, numIter=100, step=100, accelerate=True, stochastic=False, mini_batch=None, verbose=False, is_save=True, save_path='result', xtrue=None):
    """
    Plug-and-Play APGM with switch for PGM and SPGM
    
    ### INPUT:
    dObj       ~ data fidelity term, measurement/forward model
    rObj       ~ regularizer term
    numIter    ~ total number of iterations
    accelerate ~ use APGM or PGM
    stochastic ~ use SPGM or not
    mini_batch ~ the size of the mini_batch
    step       ~ step-size
    verbose    ~ if true print info of each iteration
    is_save    ~ if true save the reconstruction of each iteration
    save_path  ~ the save path for is_save
    xtrue      ~ the ground truth of the image, for tracking purpose

    ### OUTPUT:
    x     ~ reconstruction of the algorithm
    outs  ~ detailed information including cost, snr, step-size and time of each iteration

    """
    
    ##### HELPER FUNCTION #####

    evaluateSnr = lambda xtrue, x: 20*np.log10(np.linalg.norm(xtrue.flatten('F'))/np.linalg.norm(xtrue.flatten('F')-x.flatten('F')))
    evaluateTol = lambda x, xnext: np.linalg.norm(x.flatten('F')-xnext.flatten('F'))/np.linalg.norm(x.flatten('F'))


    ##### INITIALIZATION #####
    
    # initialize save foler
    if save_path:
        abs_save_path = os.path.abspath(save_path)
        if os.path.exists(save_path):
            print("Removing '{:}'".format(abs_save_path))
            shutil.rmtree(abs_save_path, ignore_errors=True)
        # make new path
        print("Allocating '{:}'".format(abs_save_path))
        os.makedirs(abs_save_path)
    
    # initialize measurement mask
    if stochastic:
        totNum = dObj.y.shape[0]
        idx = 0
        keepIdx = np.zeros(mini_batch, dtype=np.int32)
        for j in range(mini_batch):
            keepIdx[j] = idx
            idx = idx + int(totNum/mini_batch)

    #initialize info data
    if xtrue is not None:
        xtrueSet = True
        snr = []
    else:
        xtrueSet = False

    dist = []
    timer = []
    relativeChange = []
    
    # initialize variables
    xinit = np.zeros(dObj.sigSize, dtype=np.float32) 
    # outs = struct(xtrueSet)
    x = xinit
    s = x            # gradient update
    t = 1.           # controls acceleration
    p = rObj.init()  # dual variable for TV
    pfull = rObj.init()  # dual variable for TV
    

    ##### MAIN LOOP #####

    for indIter in range(numIter):
        timeStart = time.time()
        if stochastic:
            # get gradient
            g, _, keepIdx = dObj.gradStoc(s, keepIdx)
            # denoise
            xnext, p = rObj.prox(np.clip(s-step*g,0,np.inf), step, p)   # clip to [0, inf]
        else:
            # get gradient
            g, _ = dObj.grad(s)
            # denoise
            xnext, p = rObj.prox(np.clip(s-step*g,0,np.inf), step, p)   # clip to [0, inf]

        # calculate full gradient 
        if stochastic:
            gfull, _ = dObj.grad(s)
            Px, pfull = rObj.prox(np.clip(s-step*gfull,0,np.inf), step, pfull)
        else:
            Px = xnext

        # if indIter == 0:
        #     outs.dist0 = np.linalg.norm(x.flatten('F') - Px.flatten('F'))^2
        
        # acceleration
        if accelerate:
            tnext = 0.5*(1+np.sqrt(1+4*t*t))
        else:
            tnext = 1
        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # output info
        # cost[indIter] = data
        dist.append(np.linalg.norm(x.flatten('F') - Px.flatten('F'))**2)
        timer.append(time.time() - timeStart)
        if indIter == 0:
            relativeChange.append(np.inf)
        else:
            relativeChange.append(evaluateTol(x, xnext))
        # evaluateTol(x, xnext)
        if xtrueSet:
            snr.append(evaluateSnr(xtrue, x))

        # update
        t = tnext
        x = xnext

        # save & print
        if is_save:
            util.save_mat(xnext, abs_save_path+'/iter_{}_mat.mat'.format(indIter+1))
            util.save_img(xnext, abs_save_path+'/iter_{}_img.tif'.format(indIter+1))
        
        if verbose:
            if xtrueSet:
                print('[fistaEst: '+str(indIter+1)+'/'+str(numIter)+']'+'[tols: %.5e]'%(relativeChange[indIter])+'[||x-Px||^2: %.5e]'%(dist[indIter])+'[step: %.1e]'%(step)+'[time: %.1f]'%(np.sum(timer))+'[snr: %.2f]'%(snr[indIter]))
            else:
                print('[fistaEst: '+str(indIter+1)+'/'+str(numIter)+']'+'[tols: %.5e]'%(relativeChange[indIter])+'[||x-Px||^2: %.5e]'%(dist[indIter])+'[step: %.1e]'%(step)+'[time: %.1f]'%(np.sum(timer)))

        # summarize outs
        #time.sleep(0.01)
        outs = {
            # 'cost': cost,
            'dist': dist,
            'time': timer,
            'relativeChange': relativeChange,
            'snr': snr
        }

    return x, outs
