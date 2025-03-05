#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:52:59 2019

@author: reubendo
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

# import niftynet.engine.logging as logging
import numpy as np
import tensorflow as tf
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.activation import ActiLayer
from niftynet.layer.bn import InstanceNormLayer
from niftynet.layer.convolution import ConvLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.downsample import DownSampleLayer as Pooling
from niftynet.layer.linear_resize import LinearResizeLayer


MODALITIES = ['T1', 'T1c', 'T2', 'Flair', 'seg']

HIDDEN_SPACE = 512

NB_CONV = 8
tf.set_random_seed(1)


class U_HVED(TrainableLayer):
    """
    Implementation of U-MVAE introduced [1] mixing MVAE [2] and a U-Net architecture [3]
    [1] Dorent, et al. "Hetero-Modal Variational Encoder-Decoder for
        Joint Modality Completion and Segmentation". 
        MICCAI 2019. 
    [2] Wu, et al. "Multimodal Generative Models for Scalable Weakly-Supervised Learning"
        NIPS 2018. https://arxiv.org/abs/1802.05335
    [3] Ronneberger, et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation". 
        MICCAI 2015. https://arxiv.org/abs/1505.04597
    """

    def __init__(self,
                num_classes,
                w_initializer=None,
                w_regularizer=None,
                b_initializer=None,
                b_regularizer=None,
                acti_func='leakyrelu',
                name='U_HVED'):

        super(U_HVED, self).__init__(name='VAE')

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        self.num_classes = num_classes

    def layer_op(self, images, choices, is_training=True, is_inference=False, **unused_kwargs):

        encoder = ConvEncoder(w_initializer=self.initializers['w'], w_regularizer=self.regularizers['w'], b_initializer=self.initializers['b'],b_regularizer=self.regularizers['b'])

        approximate_sampler = GaussianSampler()

        img_decoder = ConvDecoderImg(w_initializer=self.initializers['w'], w_regularizer=self.regularizers['w'], b_initializer=self.initializers['b'],b_regularizer=self.regularizers['b'])

        mod_img = MODALITIES[:4]

        # Encode the input
        post_param = encoder(images)

        # Sample from the posterior distribution P(latent variables|input)
        skip_flow = []
        for k in range(len(post_param)):
            sample = approximate_sampler(post_param[k]['mu'], post_param[k]['logvar'], mod_img, choices,  is_inference)
            skip_flow.append(sample)

        img_output = img_decoder(skip_flow)

        return img_output, post_param


class ConvEncoder(TrainableLayer):
    """
    Each modality are encoded indepedently.
    """

    def __init__(self,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='ConvEncoder'):

        super(ConvEncoder, self).__init__(name=name)

        #self.denoising_variance = denoising_variance

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        self.ini_f = NB_CONV
        self.layers = [
            {'name': 'conv_0', 'n_features': self.ini_f, 'kernel_size': (1,1,1)},
            {'name': 'block_1', 'n_features': self.ini_f, 'kernels': ((3,3,3), (3,3,3)), 'downsampling':True},
            {'name': 'block_2', 'n_features': 2*self.ini_f, 'kernels': ((3,3,3), (3,3,3)), 'downsampling':True},
            {'name': 'block_3', 'n_features': 4*self.ini_f, 'kernels': ((3,3,3), (3,3,3)), 'downsampling':True},
            {'name': 'block_4', 'n_features': 8*self.ini_f, 'kernels': ((3,3,3), (3,3,3)), 'downsampling':False}]

        self.skip_ind = [1, 3, 5, 7]
        self.hidden = [self.layers[k]['n_features'] for k in range(1,len(self.layers))] 
        self.hidden = [int(k/2) for k in self.hidden]
        

    def layer_op(self, images):
        # Define the encoding convolutional layers
        def clip(input):
            # This is for clipping logvars,
            # so that variances = exp(logvars) behaves well
            output = tf.maximum(input, -50)
            output = tf.minimum(output, 50)
            return output
        
        layer_instances = [] #list layers
        means = dict()
        logvars = dict()

        pooling_params = {'kernel_size': 2, 'stride': 2}


        list_skip_flow = [{'mu':dict(), 'logvar':dict()} for k in range(len(self.skip_ind))]

        layer_fc_mod = dict()
        layer_cnn_mod = dict()
        

        for mod in MODALITIES[:4]:
            layer_cnn_mod[mod] = []
            layer_fc_mod[mod] = []
            params = self.layers[0]
            first_conv_layer = ConvolutionalLayer(
                n_output_chns=params['n_features'],
                kernel_size=params['kernel_size'],
                acti_func='leakyrelu',
                with_bn=False,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='%s_%s' % (params['name'],mod))
            layer_instances.append(first_conv_layer)
            layer_cnn_mod[mod].append(first_conv_layer)

            for i in range(1,len(self.layers)):
                params = self.layers[i]
                res_block = ResBlock(
                    n_output_chns=params['n_features'],
                    kernels=params['kernels'],
                    acti_func='leakyrelu',
                    encoding=True,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%s' % (params['name'],mod))
                layer_instances.append(res_block)
                layer_cnn_mod[mod].append(res_block)
                
                if params['downsampling']:    
                    downsampler = Pooling(func='MAX', kernel_size=2, stride=2,)
                    layer_instances.append(downsampler)
                    layer_cnn_mod[mod].append(downsampler)

        print(MODALITIES[:4])
        for mod in MODALITIES[:4]:
            flow_mod = images[mod]
            print(flow_mod)
            
            for ind, cnn_mod in enumerate(layer_cnn_mod[mod]):
                flow_mod = cnn_mod(flow_mod)
                layer_cnn_mod[mod][ind] = cnn_mod
                if ind in self.skip_ind:
                    pos = self.skip_ind.index(ind)
                    list_skip_flow[pos]['mu'][mod] = flow_mod[...,:self.hidden[pos]]
                    list_skip_flow[pos]['logvar'][mod] = clip(flow_mod[...,self.hidden[pos]:])

        output = list_skip_flow

        self._print(layer_instances)
        return output


    def _print(self, list_of_layers):
        for op in list_of_layers:
            print(op)

class GaussianSampler(TrainableLayer):
    """
        This predicts the mean and logvariance parameters,
        then generates an approximate sample from the posterior.
    """

    def __init__(self,
                 name='gaussian_sampler'):

        super(GaussianSampler, self).__init__(name=name)


    def layer_op(self, means, logvars, list_mod, choices,  is_inference):
        mu_prior = tf.zeros(tf.shape(means[list_mod[0]]))
        log_prior = tf.zeros(tf.shape(means[list_mod[0]]))

        eps=1e-7
        T = tf.boolean_mask([1/(tf.exp(logvars[mod]) + eps)  for mod in list_mod], choices)
        mu = tf.boolean_mask([means[mod]/(tf.exp(logvars[mod]) + eps) for mod in list_mod], choices)

        T = tf.concat([T,[1+log_prior]], 0)
        mu = tf.concat([mu,[mu_prior]], 0)

        posterior_means = tf.reduce_sum(mu,0) / tf.reduce_sum(T,0)
        var = 1 / tf.reduce_sum(T,0)
        posterior_logvars = tf.log(var + eps)

        if is_inference:
            return posterior_means
        else:
            noise_sample = tf.random_normal(tf.shape(posterior_means),
                                                0.0,
                                                1.0)
            output = posterior_means + tf.exp(0.5 * posterior_logvars) * noise_sample
            return output


class ConvDecoderImg(TrainableLayer):
    """
    Each modality are decoded indepedently using the multi-scale hidden samples.
    """

    def __init__(self,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='ConvDecoderImg'):

        super(ConvDecoderImg, self).__init__(name=name)


        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        self.ini_f = NB_CONV
        self.layers = [
            {'name': 'block_1', 'n_features': 4*self.ini_f, 'kernels': ((3,3,3), (3,3,3))},
            {'name': 'block_2', 'n_features': 2*self.ini_f, 'kernels': ((3,3,3), (3,3,3))},
            {'name': 'block_3', 'n_features': self.ini_f, 'kernels': ((3,3,3), (3,3,3))}]

    def layer_op(self, list_skips):

        # Define the decoding convolutional layers
        layer_instances = [] #list layers
        layer_mod = dict()
        decoders_fc = dict()
        flow = dict()

        list_skips = list_skips[::-1]

        for mod in MODALITIES:
            layer_mod[mod] = []

            flow_mod = list_skips[0]

            if mod =='seg':
                double = True
                n_output = 4
            else:
                double = True
                n_output = 1

            for i in range(len(self.layers)):
                
                params = self.layers[i]

                flow_mod = LinearResizeLayer(list_skips[i+1].shape.as_list()[1:-1])(flow_mod)

                print(mod)
                print(flow_mod)
                print('added with ')
                print(list_skips[i+1])
                flow_mod = ElementwiseLayer('CONCAT')(flow_mod, list_skips[i+1])
                print(flow_mod)


                res_block = ResBlock(
                    n_output_chns=params['n_features'],
                    kernels=params['kernels'],
                    acti_func='leakyrelu',
                    encoding=False,
                    double_n = double,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%s' % (params['name'],mod))
                layer_instances.append(res_block)
                layer_mod[mod].append(res_block)
                flow_mod = res_block(flow_mod)


            last_conv = ConvolutionalLayer(
                n_output_chns=n_output,
                kernel_size=(1,1,1),
                with_bn=False,
                acti_func=None,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='final_conv_seg')
            layer_instances.append(last_conv)
            flow_mod = last_conv(flow_mod)
            flow[mod] = flow_mod

        self._print(layer_instances)
        return flow
    
    def _print(self, list_of_layers):
        for op in list_of_layers:
            print(op)

class ResBlock(TrainableLayer):
    """
    This class define a high-resolution block with residual connections
    kernels

        - specify kernel sizes of each convolutional layer
        - e.g.: kernels=(5, 5, 5) indicate three conv layers of kernel_size 5

    with_res

        - whether to add residual connections to bypass the conv layers
    """

    def __init__(self,
                 n_output_chns,
                 kernels=((3,3,3), (3,3,3)),
                 acti_func='leakyrelu',
                 encoding=False,
                 double_n = True,
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='ResBlock',
                 stride=1):

        super(ResBlock, self).__init__(name=name)

        self.n_output_chns = n_output_chns
        if hasattr(kernels, "__iter__"):  # a list of layer kernel_sizes
            self.kernels = kernels
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
        self.acti_func = acti_func
        self.with_res = with_res

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}
        self.stride = stride
        self.encoding = encoding
        self.double_n = double_n
        self.kernels = self.kernels if double_n else [self.kernels[0]]

    def layer_op(self, input_tensor):
        output_tensor = input_tensor
        for (i, k) in enumerate(self.kernels):
            # create parameterised layers
            if self.encoding:
                if i==0:
                    nb_channels = self.n_output_chns
                elif i==1:
                    nb_channels = self.n_output_chns
            else:
                if self.double_n:
                    if i==0:
                        nb_channels = self.n_output_chns
                    elif i==1:
                        nb_channels = int(self.n_output_chns/2)
                else:
                    nb_channels = int(self.n_output_chns/2)

            in_op = InstanceNormLayer(name='in_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_func,
                                regularizer=self.regularizers['w'],
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=nb_channels,
                                kernel_size=k,
                                stride=self.stride,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                name='conv_{}'.format(i))
            # connect layers
            output_tensor = in_op(output_tensor)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)
        return output_tensor


