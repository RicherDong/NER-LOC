# -*- coding: utf-8 -*-
# @Time   : 2018/11/12 15:46
# @Author : Richer
# @File   : IDCNN_model.py

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from model.model_base import model_base

import sys, os
sys.path.append('../config/')


class cnn_model(model_base):

    def __init__(self, config):
        super().__init__(config)
        self.embedding_dim = self.char_dim + self.seg_dim  # 120  # 字的维度 + 切词信息的维度 = 实际维度
        self.num_filter    = self.lstm_dim  # 100 双向lstm 是卷积核的个数
        self.get_layers()
        model_outputs = self.IDCNN_layer()
        self.logits = self.full_connect_idcnn_layer(model_outputs)
        self.loss   = self.loss_layer(self.logits, self.lengths)
        self.get_optimizer()



    def get_layers(self):
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]



    def IDCNN_layer(self):
        """
        使用膨胀卷积网络
        :param idcnn_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, cnn_output_width]
        """
        # tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置（维度从0开始）。一般4维
        model_inputs = tf.expand_dims(self.embedding, 1)
        reuse = True  if self.dropout == 1.0  else False
        with tf.variable_scope("idcnn"):
            filter_weights = tf.get_variable(name="idcnn_filter",
                                             shape=[1, self.filter_width,  self.embedding_dim, self.num_filter],  #对应的参数含义: 高度, 宽度, 通道数(输入通道数), 卷积核个数(输出通道数)  字一般是没有高度的, 只有宽度, 通道数一般和input的第四维是一致的
                                             initializer=self.initializer)

            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,      # [batch_size, height, weight, channel]
                                      filter_weights,    # 此位置文字的卷积核 [高, 宽, 输入通道, 输出通道数]
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer",
                                      use_cudnn_on_gpu=True)
            # 先卷积一次 提取特征
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            layer_num = len(self.layers)  # 3层
            for j in range(self.repeat_times):              # 分四个分支进行重复的训练
                for i in range(layer_num):                  # 此位置遍历的是膨胀系数   # 1,1,2 膨胀系数dilation 是1相当于没有膨胀 padding="SAME",是一样的。
                    dilation = self.layers[i]['dilation']   # 去除膨胀系数
                    isLast = True if i == (layer_num - 1) else False    # 当i等于2的时候是最后一次卷积
                    with tf.variable_scope("atrous-conv-layer-%d" % i,  # 相同作用于下, 膨胀卷积作用参数可以共用
                                           reuse=True  if (reuse or j > 0) else False):  # 节省空间 # 在variable_scope作用域下都是True 也就是每次w,b,都会复用，用的是同一个变量
                        w = tf.get_variable(name="filter_w",
                                            shape=[1, self.filter_width, self.num_filter, self.num_filter],       # w 卷积核的高度，卷积核的宽度， 图像通道数，卷积核个数
                                            initializer=self.initializer)   # 初始化卷积和
                        b = tf.get_variable(name= "filterB", shape=[self.num_filter])  # 初始化偏值
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")  # 膨胀卷积
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        layerInput = conv
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)  # 4层信息都放进来
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            finalOut = tf.squeeze(finalOut, [1])  # 把添加的维度去掉 宽，高不变
            # <tf.Tensor 'idcnn/Squeeze:0' shape=(?, ?, 400) dtype=float32>
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])  # 相当于拉直维度
            # 把前两个维度乘起来，最后面的维度不变
            # <tf.Tensor 'idcnn/Reshape:0' shape=(?, 400) dtype=float32>
            self.cnn_output_width = totalWidthForLastDim  # 400
            return finalOut                               # 特征提取结束


    def full_connect_idcnn_layer(self, idcnn_outputs):
        """
        全连接层
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"):
            with tf.variable_scope("logits"):
                W = tf.get_variable(name = "w", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])   # num_steps 一句话中字的个数， num_tags 是对应的tags 数   这一部分就是进行reshape成原类型进行对比

