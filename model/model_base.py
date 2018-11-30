# -*- coding: utf-8 -*-
# @Time   : 2018/11/13 11:45
# @Author : Mat
# @File   : model_base.py

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import numpy as np


class model_base(object):

    def __init__(self, config):
        self.char_dim   = config.char_dim    # 100 字的id
        self.lstm_dim   = config.lstm_dim    # 100字的向量
        self.seg_dim    = config.seg_dim     # 20 切词的id
        self.num_tags   = config.num_tags    #51
        self.num_chars  = config.num_chars   #样本中总字数 3538
        self.steps_check= config.steps_check # 检查频率
        self.num_segs   = config.num_segs
        self.filter_width = config.filter_width  # 卷积和的大小
        self.repeat_times = config.repeat_times  # 使用膨胀卷积卷积次数
        self.optimizer    = config.optimizer
        self.clip         = config.clip
        self.learning_rate= config.learn_rate
        self.loss         = ''
        self.cnn_output_width = 0  # 输出的宽度为0
        self.initializer = initializers.xavier_initializer()  # 正态分布的方法
        self.get_variable_placeholder()
        self.get_lengths()
        self.embedding = self.embedding_layer(self.char_inputs, self.seg_inputs)


    def get_variable_placeholder(self):
        # 定义变量
        self.global_step = tf.Variable(0,   trainable=False)
        # 定义占位符
        self.char_inputs  = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ChatInputs")
        self.seg_inputs   = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")
        self.targets      = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        self.dropout      = tf.placeholder(dtype=tf.float32, name="Dropout")


    def get_lengths(self):
        used   = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps  = tf.shape(self.char_inputs)[-1]


    def embedding_layer(self, char_inputs, seg_inputs):
        """
        此方法的目的是将字向量与分词向量进行合并
        :param char_inputs: one-hot encoding of sentence  输入数据
        :param seg_inputs: segmentation feature  嵌入的分词的信息
        :return: [1, num_ste ps, embedding size],
        """
        embedding = []
        embedding_append = embedding.append
        # self.char_inputs_test= char_inputs
        # self.seg_inputs_test = seg_inputs
        with tf.variable_scope("char_embedding"):
            self.char_lookup = tf.get_variable(name = "char_embedding",
                                               shape = [self.num_chars, self.char_dim],   # 初始化字向量， 转换为 [文本中总字数, 100]   100 维度的字向量
                                               initializer=self.initializer)
            embedding_append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))

            if self.seg_dim:       # 存在20 维度的定义
                with tf.variable_scope("seg_embedding"):
                    self.seg_lookup = tf.get_variable(name="seg_embedding",
                                                      shape=[self.num_segs, self.seg_dim],  #num_segs分词的状态信息seg_dim字对应的id   4 * 20 维的向量
                                                      initializer=self.initializer)
                    embedding_append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))  #每个字都有四种状态
            embed = tf.concat(embedding, axis=-1)    #按最后一个维度拼接 bitch_size 句子中字的个数
        return embed


    def loss_layer(self, project_logits, lengths):
        """
         最终的特征采用条件随机解码器
         条件随机 看上下文 预测下个字的tag 看特征函数，上个字，状态转移，上个解码tag是啥概率是多大
         SoftMax不看上下文
         来算 project_logits 特征  lengths句子的解码长度
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags] num_steps字的个数
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            small = -1000.0    #定义看状态，看特征
            start_logits = tf.concat([small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)  #batch_size*1*(num_tags+1)初始化
            #初始化一些 1
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)  #self.num_steps 是代表的是字的长度
            logits = tf.concat([project_logits, pad_logits], axis=-1)                                #最后一维concat project_logits=51+1   52
            logits = tf.concat([start_logits, logits], axis=1)                                       #合并[a,b,c] b维上的维度上的维度
            targets = tf.concat([tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            # 特征转移矩阵，状态转移矩阵
            self.trans = tf.get_variable(name = "transitions",
                                         shape=[self.num_tags + 1, self.num_tags + 1],
                                         initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(inputs=logits,
                                                            tag_indices=targets,
                                                            transition_params=self.trans,#转移矩阵
                                                            sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)


    def get_optimizer(self):
        '''
        选择优化器
        :return:
        '''
        with tf.variable_scope("optimizer"):
            self.lr = tf.train.exponential_decay(self.learning_rate,
                                            self.global_step, 15000, 0.99, staircase=True)
            optimizer = self.optimizer        # adam
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            grads_vars = self.opt.compute_gradients(self.loss)  # len(grads_vars) 12

            capped_grads_vars = [[tf.clip_by_value(g, -self.clip, self.clip), v]  for g, v in grads_vars]  # 梯度进行截断（更新）
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)  # global_step要求解的一个值
