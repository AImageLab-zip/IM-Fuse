"""
@author: reubendo
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
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
class U_HeMIS(TrainableLayer):
    """
    Implementation of U-HeMIS introduced [1] mixing HeMIS [2] and a U-Net architecture [3]
    [1] Dorent, et al. "Hetero-Modal Variational Encoder-Decoder for
        Joint Modality Completion and Segmentation". 
        MICCAI 2019.
    [2] Havaei, et al. "HeMIS: Hetero-Modal Image Segmentation". 
        MICCAI 2016. https://arxiv.org/abs/1607.05194
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
                name='VAE'):

        super(U_HeMIS, self).__init__(name=name)

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        self.num_classes = num_classes

    def layer_op(self, images, choices, is_training=True, is_inference=False, **unused_kwargs):

        encoder = ConvEncoder(w_initializer=self.initializers['w'], w_regularizer=self.regularizers['w'], b_initializer=self.initializers['b'],b_regularizer=self.regularizers['b'])

        abstraction_op = HeMISAbstractionBlock()

        img_decoder = ConvDecoderImg(w_initializer=self.initializers['w'], w_regularizer=self.regularizers['w'], b_initializer=self.initializers['b'],b_regularizer=self.regularizers['b'])


        mod_img = MODALITIES[:4]

        # Encode the input
        list_skips = encoder(images)

        # Sample from the posterior distribution P(latent variables|input)
        skip_flow = []
        for k in range(len(list_skips)):
            sample = abstraction_op(list_skips[k],  choices,  is_inference)
            skip_flow.append(sample)

        img_output = img_decoder(skip_flow)

        return img_output


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


        list_skip_flow = [[] for k in range(len(self.skip_ind))]

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




        
        for mod in MODALITIES[:4]:
            flow_mod = images[mod]
            print(flow_mod)
            
            for ind, cnn_mod in enumerate(layer_cnn_mod[mod]):
                
                flow_mod = cnn_mod(flow_mod)
                layer_cnn_mod[mod][ind] = cnn_mod
                if ind in self.skip_ind:
                    pos = self.skip_ind.index(ind)
                    list_skip_flow[pos].append(flow_mod)
                print(flow_mod)
            print('list_flow')
        
        print(list_skip_flow)
            

        output = list_skip_flow



        if True:
            self._print(layer_instances)
            return output
        return output

    def _print(self, list_of_layers):
        for op in list_of_layers:
            print(op)

class HeMISAbstractionBlock(TrainableLayer):
    def __init__(self,
                 pooling_type='average',
                 name='HeMISAbstractionBlock'):

        super(HeMISAbstractionBlock, self).__init__(name=name)

        self.pooling_type = pooling_type
        self.name = name

    def layer_op(self, input_tensor, choices, is_training):
        """
        Written by Thomas Varsavsky.

        Function will drop all zero columns and compute E[C] and Var[C]
        :param backend_output: backend_output
        :return: 1xC tensor where C is the number of features.
        """
        # Omit zero columns from average
        # intermediate_tensor = tf.reduce_sum(tf.abs(input_tensor), 0)
        # zero_vector = tf.zeros(shape=(1, 1), dtype=tf.float32)
        # bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
        # omit_zero_columns = tf.boolean_mask(input_tensor, bool_mask)
        # Compute E[C]
        input_tensor = tf.boolean_mask(input_tensor, choices)
        average_over_modalities, variance_between_modalities = tf.nn.moments(input_tensor, axes=[0])
        abstraction_output = tf.concat([average_over_modalities, variance_between_modalities], axis=-1)
        return abstraction_output


class ConvDecoderImg(TrainableLayer):
    """
    Each modality are then decoded using the average of the skip-connections across
    the available modalities. 
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

        for mod in ['seg']:
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

        if True:
            self._print(layer_instances)
            return flow['seg']
        return flow['seg']
    
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
                    nb_channels = int(self.n_output_chns/2)
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
        # make residual connections
        # if self.with_res:
        #     output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor


