# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the MobileNet
- MobileNet, as described inhttps://arxiv.org/abs/1704.04861
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import tensorflow.contrib.slim as slim

#######################
#MobileNet related imports
import random
import numpy as np
from math import pi
import os

#MobileNet functions
def conv2dBN(x, W, training, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.contrib.layers.batch_norm(x, is_training=training, center=True, scale=True, decay=0.9, updates_collections=None)
    return tf.nn.relu(x)

def conv2dSepBN(x, W1, W2, training, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.separable_conv2d(x, W1, W2, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.contrib.layers.batch_norm(x, is_training=training, center=True, scale=True, decay=0.9, updates_collections=None)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv2d2(x, W, training, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return tf.nn.relu(x)


#######################

'''
    # Build the tf graph
    weights_mobilenet = (tf.Variable(tf.random_normal([3, 3, inDepth, 32])), # 0
    tf.Variable(tf.random_normal([3, 3, 32, 1])),       # 1
    tf.Variable(tf.random_normal([1, 1, 32, 32])),      # 1
    tf.Variable(tf.random_normal([3, 3, 32, 1])),       # 2
    tf.Variable(tf.random_normal([1, 1, 32, 48])),      # 2
    tf.Variable(tf.random_normal([3, 3, 48, 1])),
    tf.Variable(tf.random_normal([1, 1, 48, 48])),
    tf.Variable(tf.random_normal([3, 3, 48, 1])),
    tf.Variable(tf.random_normal([1, 1, 48, 64])),
    tf.Variable(tf.random_normal([3, 3, 64, 1])),
    tf.Variable(tf.random_normal([1, 1, 64, 64])),
    tf.Variable(tf.random_normal([3, 3, 64, 1])),
    tf.Variable(tf.random_normal([1, 1, 64, 64])),
    tf.Variable(tf.random_normal([3, 3, 64, 1])),
    tf.Variable(tf.random_normal([1, 1, 64, 64])),
    tf.Variable(tf.random_normal([1, 1, 64, 1])))
    strides_mobilenet = (1, 1, 2, 1, 2, 1, 1, 1, 1)
    maxpool_mobilenet = (2, 1, 1, 1, 1, 1, 1, 1, 1)
    prediction = mobilenet(X, weights_mobilenet, strides_mobilenet, maxpool_mobilenet, is_training)
    '''
def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    #TODO: the inputs (keep_probability, bottleneck_layer_size=128, weight_decay=0.0, reuse=None are not used, but kept due to the function's signature
    
    # Build the tf graph
    weights_mobilenet = (tf.Variable(tf.random_normal([3, 3, inDepth, 32])), # 0
                     tf.Variable(tf.random_normal([3, 3, 32, 1])),       # 1
                     tf.Variable(tf.random_normal([1, 1, 32, 32])),      # 1
                     tf.Variable(tf.random_normal([3, 3, 32, 1])),       # 2
                     tf.Variable(tf.random_normal([1, 1, 32, 48])),      # 2
                     tf.Variable(tf.random_normal([3, 3, 48, 1])),
                     tf.Variable(tf.random_normal([1, 1, 48, 48])),
                     tf.Variable(tf.random_normal([3, 3, 48, 1])),
                     tf.Variable(tf.random_normal([1, 1, 48, 64])),
                     tf.Variable(tf.random_normal([3, 3, 64, 1])),
                     tf.Variable(tf.random_normal([1, 1, 64, 64])),
                     tf.Variable(tf.random_normal([3, 3, 64, 1])),
                     tf.Variable(tf.random_normal([1, 1, 64, 64])),
                     tf.Variable(tf.random_normal([3, 3, 64, 1])),
                     tf.Variable(tf.random_normal([1, 1, 64, 64])),
                     tf.Variable(tf.random_normal([1, 1, 64, 1])))
    strides_mobilenet = (1, 1, 2, 1, 2, 1, 1, 1, 1)
    maxpool_mobilenet = (2, 1, 1, 1, 1, 1, 1, 1, 1)
    return mobilenet(images, weights_mobilenet, strides_mobilenet, maxpool_mobilenet, is_training=phase_train)

def mobilenet(x, weights, strides, maxpools, is_training):
    """Creates the MobileNet model.
        Args:
        inputs: a 4-D tensor of size [batch_size, height, width, 3].
        is_training: whether is training or not.
        scope: Optional variable_scope.
        Returns:
        logits: the logits outputs of the model.
        """
    noOfLayers = len(strides)
    # build the first layer
    conv = conv2dBN(x, weights[0], is_training, strides[0])
    if( maxpools[0] > 1 ):
        conv = maxpool2d(conv, k=maxpools[0])
    # build the middle layers
    for i in range(1,noOfLayers-1):
        conv = conv2dSepBN(conv, weights[2*i-1], weights[2*i], is_training, strides[i])
        if( maxpools[i] > 1 ): conv = maxpool2d(conv, k=maxpools[i])
    # build the last layer
    # no batch normalization at the last layer of the network.
    conv = conv2d2(conv, weights[-1], is_training, strides[-1])
    print("NETWORK OUTPUT SIZE: " + str(conv.get_shape()))
    return conv, noOfLayers

