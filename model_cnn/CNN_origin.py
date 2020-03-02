#!/usr/bin/env python
#encoding=utf-8

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

class CNN(object):
    def __init__(
      self, max_input_left,embeddings,vocab_size,embedding_size,batch_size,filter_sizes,
      num_filters,dataset,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,extend_feature_dim = 10):

        self.num_filters = num_filters
        self.embeddings=embeddings
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.trainable = trainable
        self.filter_sizes = filter_sizes
        self.batch_size = batch_size
        self.dataset=dataset
        self.l2_reg_lambda = l2_reg_lambda
        self.para = []
        self.max_input_left = max_input_left
        self.extend_feature_dim = extend_feature_dim
        self.is_Embedding_Needed = is_Embedding_Needed
        self.rng = 23455
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'input_question')
        if self.dataset=='TREC':
            self.input_y = tf.placeholder(tf.float32, [self.batch_size,6], name = "input_y")
        else:
            self.input_y = tf.placeholder(tf.float32, [self.batch_size,2], name = "input_y")
        self.q_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'q_position')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    
    def density_weighted(self):
        self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]) , name = 'weighted_q')
        self.para.append(self.weighted_q)
    def density_matrix(self,sentence_matrix,sentence_weighted):
        # print sentence_matrix
        # print tf.nn.l2_normalize(sentence_matrix,2)
        self.norm = tf.nn.l2_normalize(sentence_matrix,2)
        reverse_matrix = tf.transpose(self.norm, perm = [0,1,3,2])
        q_a = tf.matmul(self.norm,reverse_matrix)
        # return tf.reduce_sum(tf.matmul(self.norm,reverse_matrix), 1)
        return tf.reduce_sum(tf.multiply(q_a,sentence_weighted),1)
    
    def add_embeddings(self):
        with tf.name_scope("embedding"):
            if self.is_Embedding_Needed:
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
                W_pos=tf.Variable(tf.random_uniform([500, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
            else:
                W_pos=tf.Variable(tf.random_uniform([500, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)                
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
            self.embedding_W = W
            self.embedding_W_pos=W_pos
        self.embedded_chars_q,self.embedded_chars_q_pos = self.concat_embedding(self.question,self.q_position)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars_q, -1)
        self.embedded_chars_expanded = self.density_matrix(self.embedded_chars_expanded,self.weighted_q)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars_expanded, -1)

    def feed_neural_work(self):
        pooled_outputs = []
        l2_loss = tf.constant(0.0)
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size,
                                self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.embedding_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            if self.dataset=='TREC':
                W = tf.get_variable("W",shape=[num_filters_total, 6],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[6]), name="b")
            else:
                W = tf.get_variable("W",shape=[num_filters_total, 2],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
    def concat_embedding(self,words_indice,position_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        embedding_chars_q_pos=tf.nn.embedding_lookup(self.embedding_W_pos,position_indice)
        return embedded_chars_q,embedding_chars_q_pos
    def build_graph(self):
        self.create_placeholder()
        self.density_weighted()
        self.add_embeddings()
        self.feed_neural_work()

