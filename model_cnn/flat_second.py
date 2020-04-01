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

class second_feature(fasttext):
    def __init__(
      self, max_input_left,embeddings,vocab_size,embedding_size,batch_size,filter_sizes,
      num_filters,dataset,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,extend_feature_dim = 10,dropout_keep_prob=0.5):

        super().__init__(max_input_left,embeddings,vocab_size,embedding_size,batch_size,filter_sizes,
      num_filters,dataset,l2_reg_lambda, is_Embedding_Needed ,trainable ,extend_feature_dim )

        self.dropout_keep_prob = dropout_keep_prob

    def density_weighted(self):

        one = [1. for i in range(self.max_input_left)]
        sentence_weight = tf.Variable(tf.diag(one))
        sentence_weight = tf.expand_dims(sentence_weight,0)
        return sentence_weight

    def density_matrix(self,embeddings,sentence_weight):
        embedded_trans = tf.transpose(embeddings,[0,2,1]) #[batc,embeddings,seq_len]
        embedded_first = tf.matmul(embedded_trans,sentence_weight)
        embedded_second = tf.matmul(embedded_first,embeddings)
        return embedded_second #[batch,embeddings,embeddings]

    def Convolution(self,input_x_images):

 
        # 进行拼接
        num_filters_total =  self.embedding_size * self.embedding_size
        h_pool_flat = tf.reshape(input_x_images, [-1, num_filters_total])
 
        # 对最终得到的句子向量进行dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        return h_drop,num_filters_total


    def build_graph(self):

        self.create_placeholder()
        embedded_chars_q = self.get_embedding(self.question)
        sentence_weight = self.density_weighted()
        density_matrix = self.density_matrix(embedded_chars_q,sentence_weight)
        hidden,num_filters_total = self.Convolution(density_matrix)


        self.clasify(hidden,num_filters_total)
    
if __name__ == '__main__':
    cnn = CNN(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 50,
                batch_size = 3,
                embeddings = None,
                embeddings_complex=None,
                dropout_keep_prob = 1,
                filter_sizes = [40],
                num_filters = 65,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                overlap_needed = False,
                pooling = 'max',
                position_needed = False)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3*33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,33))
    input_overlap_a = np.ones((3,40))
    q_posi = np.ones((3,33))
    a_posi = np.ones((3,40))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.input_y:input_y,
            cnn.q_position:q_posi,
        }

        see,question,scores = sess.run([cnn.embedded_chars_q,cnn.question,cnn.scores],feed_dict)
        print (see)

