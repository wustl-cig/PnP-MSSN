# Multiple Self-Similarity Network (MSSN) Based Plug-and-Play Prior for MRI Reconstruction
This is a Tensorflow implementation of the SPL2020 paper [A New Recurrent Plug-and-Play Prior Based on the Multiple Self-Similarity Network](https://ieeexplore.ieee.org/document/9018286)

## Abstract
Recent work has shown the effectiveness of the plug-and-play priors (PnP) framework for regularized image reconstruction. However, the performance of PnP depends on the quality of image denoiser used as a prior. In this letter, we design a novel PnP denoising prior, called multiple self-similarity net (MSSN), based on the recurrent neural network (RNN) with self-similarity matching using multi-head attention mechanism. Unlike traditional neural net denoisers, MSSN exploits different types of relationships among non-local and repeating features to remove the noise in the input image. We numerically evaluate the performance of MSSN as a module within PnP for solving magnetic resonance (MR) image reconstruction. Experimental results show the stable convergence and excellent performance of MSSN for reconstructing images from highly compressive Fourier measurements.

## Usage
```
python main.py
```
* One sample data in [fastMRI dataset](https://fastmri.org/dataset) named *'MRI_Knee_58.mat'* is in folder *'./data/'*
* The settings of plug-and-play algorithm and neural network are in file *settings.py*
* The MSSN is trained on BSD500 dataset, and the checkpoints are in folder *'./models/checkpoints/'*
* 36- and 48-line radial sampling of k-space are used in our experiments. 

## PnP-BM3D vs. PnP-DnCNN vs. PnP-MSSN
![visualExamples](images/compare.gif)

## Citation
```
@article{song2019new,
    title={A New Recurrent Plug-and-Play Prior Based on the Multiple Self-Similarity Network},
    author={Song, Guangxiao and Sun, Yu and Liu, Jiaming and Wang, Zhijie and Kamilov, Ulugbek S},
    journal={IEEE Signal Processing Letters},
    year={2020},
    doi={10.1109/TCI.2019.2893568},
}
```