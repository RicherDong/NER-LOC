# -*- coding: utf-8 -*-
# @Time   : 2018/11/26 14:12
# @Author : Richer
# @File   : BiLSTM_model.py
import os, sys;
from model.model_base import model_base
import tensorflow as tf
from tensorflow.contrib import rnn
sys.path.append('../config/')

class BiLSTM_model(model_base):

    def __init__(self,config):
        super().__init__(config)
        model_outputs = self.biLSTM_layer()
        self.logits = self.project_layer_bilstm(model_outputs)
        self.loss = self.loss_layer(self.logits, self.lengths)
        self.get_optimizer()


    def biLSTM_layer(self):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("BiLSTM"):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        self.lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                        lstm_cell["forward"],
                        lstm_cell["backward"],
                        self.embedding,
                        dtype=tf.float32,
                        sequence_length=self.lengths)
        return tf.concat(outputs, axis=2)


    def project_layer_bilstm(self, lstm_outputs):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])