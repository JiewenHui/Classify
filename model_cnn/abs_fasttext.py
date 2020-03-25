#!/usr/bin/env python
#encoding=utf-8
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from multiply import ComplexMultiply
import math
from scipy import linalg
from numpy.random import RandomState
rng = np.random.RandomState(23455)
from keras import initializers
from keras import backend as K
import math
from model_cnn.fasttext import fasttext

class abs_fasttext(fasttext):
    def __init__(
      self, max_input_left,embeddings,vocab_size,embedding_size,batch_size,filter_sizes,
      num_filters,dataset,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,extend_feature_dim = 10):

        super().__init__(max_input_left,embeddings,vocab_size,embedding_size,batch_size,filter_sizes,
      num_filters,dataset,l2_reg_lambda, is_Embedding_Needed ,trainable ,extend_feature_dim )
    def get_representation(self,embedded):
        hidden = tf.reduce_sum(embedded,-2)
        # hidden = tf.abs(hidden)
        hidden = tf.abs(hidden)
        return hidden 

    
