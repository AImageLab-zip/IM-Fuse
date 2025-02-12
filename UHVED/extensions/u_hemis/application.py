import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_weighted_v2 import WeightedSampler
from niftynet.engine.sampler_balanced_v2 import BalancedSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss_segmentation import LossFunction
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.evaluation.segmentation_evaluator import SegmentationEvaluator
from niftynet.layer.rand_elastic_deform import RandomElasticDeformationLayer
import pandas as pd
from itertools import chain, combinations
import random
import numpy as np
from multi_modal_application import MultiModalApplication


SUPPORTED_INPUT = set(['image', 'label', 'weight', 'sampler', 'inferred', 'choices', 'output_mod'])
MODALITIES_img = ['T1', 'T1c', 'T2', 'Flair']

np.random.seed(0)
tf.set_random_seed(1)

def str2bool(v):
  return v.lower() in ("true")


class U_HeMISApplication(MultiModalApplication):
    REQUIRED_CONFIG_SECTION = "MULTIMODAL"

    def __init__(self, net_param, action_param, action):
        super(MultiModalApplication, self).__init__()
        tf.logging.info('starting segmentation application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.segmentation_param = None
        self.SUPPORTED_SAMPLING = {
            'uniform': (self.initialise_uniform_sampler,
                        self.initialise_grid_sampler,
                        self.initialise_grid_aggregator),
            'weighted': (self.initialise_weighted_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
            'resize': (self.initialise_resize_sampler,
                       self.initialise_resize_sampler,
                       self.initialise_resize_aggregator),
            'balanced': (self.initialise_balanced_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
        }

    def set_iteration_update(self, iteration_message):


        if self.is_training:
            choices = []
            nb_choices = np.random.randint(4)
            choices = np.random.choice(4, nb_choices+1, replace=False, p=[1/4,1/4,1/4,1/4])
            choices = [True if k in choices else False for k in range(4)]
            print(choices)
            iteration_message.data_feed_dict[self.choices] = choices

            n_iter = iteration_message.current_iter
            decay = 4
            leng = 10000
            iteration_message.data_feed_dict[self.lr] = self.action_param.lr /(decay**int( n_iter/leng ))


    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        # def data_net(for_training):
        #    with tf.name_scope('train' if for_training else 'validation'):
        #        sampler = self.get_sampler()[0][0 if for_training else -1]
        #        data_dict = sampler.pop_batch_op()
        #        image = tf.cast(data_dict['image'], tf.float32)
        #        return data_dict, self.net(image, is_training=for_training)

        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()


        self.var = tf.placeholder_with_default(0, [], 'var')
        self.choices = tf.placeholder_with_default([True, True, True, True], [4], 'choices')
        

        

        if self.is_training:
            self.lr = tf.placeholder_with_default(self.action_param.lr, [], 'learning_rate')
                        # if self.action_param.validation_every_n > 0:
            #    data_dict, net_out = tf.cond(tf.logical_not(self.is_validation),
            #                                 lambda: data_net(True),
            #                                 lambda: data_net(False))
            # else:
            #    data_dict, net_out = data_net(True)
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)


            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.lr)


            image = tf.cast(data_dict['image'], tf.float32)
            image_unstack = tf.unstack (image, axis=-1)

            net_seg= self.net({MODALITIES_img[k]: tf.expand_dims(image_unstack[k],-1) for k in range(4)}, self.choices, is_training=self.is_training)

            cross = LossFunction(
                n_class=4,
                loss_type='CrossEntropy')

            dice = LossFunction(
                n_class=4,
                loss_type='Dice',
                softmax=True)


            gt =  data_dict['label']
            loss_cross = cross(prediction=net_seg,ground_truth=gt, weight_map=None)
            loss_dice = dice(prediction=net_seg,ground_truth=gt)
            data_loss = loss_cross + loss_dice
    

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.reduce_mean(
                [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
            loss = data_loss + reg_loss


            grads = self.optimiser.compute_gradients(loss,  colocate_gradients_with_ops=False)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=data_loss, name='loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=loss, name='loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=self.choices, name='choices',
                average_over_devices=False, collection=CONSOLE)                  


        elif self.is_inference:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            image = tf.unstack (image, axis=-1)

            choices = self.segmentation_param.choices
            choices = [str2bool(k) for k in choices]
            print(choices)

            net_seg = self.net({MODALITIES_img[k]: tf.expand_dims(image[k],-1) for k in range(4)}, choices, is_training=True, is_inference=True)

            print('output')
            post_process_layer = PostProcessingLayer(
                    'ARGMAX', num_classes=4)
            net_seg = post_process_layer(net_seg)
            outputs_collector.add_to_collection(
                var=net_seg, name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            self.initialise_aggregator()

    def interpret_output(self, batch_output):
        if self.is_inference:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        return True

    def initialise_evaluator(self, eval_param):
        self.eval_param = eval_param
        self.evaluator = SegmentationEvaluator(self.readers[0],
                                               self.segmentation_param,
                                               eval_param)

    def add_inferred_output(self, data_param, task_param):
        return self.add_inferred_output_like(data_param, task_param, 'label')
