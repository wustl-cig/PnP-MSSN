from __future__ import print_function, division, absolute_import, unicode_literals
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import importlib
import math
import settings

opt = settings.opt
############## Basis Class ##############

class RegularizerClass(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prox(self,z,step,pin):
        pass

    @abstractmethod
    def eval(self,z,step,pin):
        pass

def img_uint8_255(data):
    return (data*255.).astype(np.uint8)

def img_float32_1(data):
    return data/255.

############## Regularizer Class ##############

class MultiHeadRNN(RegularizerClass):

    def __init__(self, sigSize, sigma=5):
        super(MultiHeadRNN, self).__init__()

        self.nx, self.ny = sigSize[0], sigSize[1]

        tf.reset_default_graph()
        # net settings
        if sigma == 5:
            self.model_checkpoints = opt.model_checkpoints

        self.model = importlib.import_module('models.' + opt.model_name)
        self.stride = 7

        self.input_image = tf.placeholder(tf.float32, shape=(None, None, None))
        self.input_image_shape = tf.shape(self.input_image)
        self.input_image_reshaped = tf.reshape(self.input_image,
                                               [self.input_image_shape[0], self.input_image_shape[1],
                                                self.input_image_shape[2], 1])
        with tf.variable_scope("model"):
            self.output_image = self.model.build_model(model_input=self.input_image_reshaped,
                                                       state_num=opt.state_num,
                                                       is_training=False)
        # init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()

        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init_local)
        if tf.gfile.Exists(self.model_checkpoints) or tf.gfile.Exists(self.model_checkpoints + '.index'):
            saver.restore(self.sess, self.model_checkpoints)
            print('Model restored from ' + self.model_checkpoints)
        else:
            print('Model not found')
            exit()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

    def _get_vars(self):
        lst_vars = []
        for v in tf.global_variables():
            lst_vars.append(v)
        return lst_vars

    def init(self):
        p = np.zeros([self.nx, self.ny])
        return p

    def prox(self, s, step, pin, prob=1., phase=False, normalize=False):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param s: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction
        """

        if len(s.shape) == 2:
            # normalize
            if normalize:
                smin = np.amin(s)
                smax = np.amax(s)
                s = (s - smin) / (smax - smin)

            s = img_uint8_255(s).astype(np.float32)
            # reshape

            nx = s.shape[0]
            ny = s.shape[1]


            noisy_img = s

            noisy_image_list = [noisy_img]
            noisy_image_patch_list = []

            h_idx_list = list(range(0, noisy_img.shape[0] - opt.patch_size, self.stride)) + [
                noisy_img.shape[0] - opt.patch_size]
            w_idx_list = list(range(0, noisy_img.shape[1] - opt.patch_size, self.stride)) + [
                noisy_img.shape[1] - opt.patch_size]
            patch_list = []
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    patch_list.append(
                        noisy_img[h_idx:h_idx + opt.patch_size, w_idx:w_idx + opt.patch_size])
            noisy_image_patch_list.append(np.stack(patch_list, axis=0))

            for i in range(len(noisy_image_list)):
                batch_no = int(math.ceil(noisy_image_patch_list[i].shape[0] / float(opt.batch_size)))
                output_img_patch_list = []
                for batch_id in range(batch_no):
                    cur_batch = noisy_image_patch_list[i][
                                batch_id * opt.batch_size:min(
                                    batch_id * opt.batch_size + opt.batch_size,
                                    noisy_image_patch_list[i].shape[0]), ...]
                    output_batch = self.sess.run(self.output_image, feed_dict={self.input_image: cur_batch})
                    output_img_patch_list.append(output_batch)

                output_img_patches = np.concatenate(output_img_patch_list, axis=0)
                h_idx_list = list(range(0, noisy_image_list[i].shape[0] - opt.patch_size, self.stride)) \
                             + [noisy_image_list[i].shape[0] - opt.patch_size]
                w_idx_list = list(range(0, noisy_image_list[i].shape[1] - opt.patch_size, self.stride)) \
                             + [noisy_image_list[i].shape[1] - opt.patch_size]

                cnt_map = np.zeros_like(noisy_image_list[i])
                output_img = np.zeros_like(noisy_image_list[i])
                cnt = 0
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        output_img[h_idx:h_idx + opt.patch_size,
                        w_idx:w_idx + opt.patch_size] += output_img_patches[cnt, :, :, :].squeeze()
                        cnt_map[h_idx:h_idx + opt.patch_size, w_idx:w_idx + opt.patch_size] += 1
                        cnt += 1
                output_img /= cnt_map
                denoised_img = output_img.squeeze() + noisy_image_list[i]

                x = np.reshape(denoised_img, [nx, ny])

                if normalize:
                    x = (smax - smin) * x + smin

                return img_float32_1(x), pin

    def eval(self,z,step,pin):
        pass