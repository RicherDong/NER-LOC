# -*- coding: utf-8 -*-
# @Time   : 2018/11/12 15:46
# @Author : Richer
# @File   : run.py
from config.config import Config
from tensorflow.contrib.crf import viterbi_decode
from flask import request
from loader import Loader
import tensorflow as tf
from model.IDCNN_model import cnn_model
from model.BiLSTM_model import BiLSTM_model
import numpy as np
import random
from util import Util
from flask import Flask
import os, sys, argparse
import json


class Run(object):

    def __init__(self, type='train', model='IDCNN'):
        self.config = Config()
        self.saver = None
        self.util = Util()
        self.loader = Loader()
        self.model_type = model
        self.logger = self.util.get_logger(self.config.log_file)
        self.model = cnn_model(self.config) if self.model_type == 'IDCNN' else BiLSTM_model(self.config)
        self.ckpt_path = self.config.cnn_ckpt_path if self.model_type == 'IDCNN' else self.config.lstm_ckpt_path
        if type == 'train':
            self.train()



    def save_model(self, sess, epoch):
        self.saver = tf.train.Saver()
        self.saver.save(sess, self.ckpt_path + '-' + str(epoch))
        self.logger.info('save model done')



    def _data_preprocess(self, data, zeros, sign=False, char_to_id=[], tag_to_id=[]):

        id_to_tag, id_to_char = [], []
        train_sentence = self.loader.load_sentences(data, zeros)
        print('数据总长度：{}'.format(len(train_sentence)))
        self.loader.update_tag_schema(train_sentence, self.config.tag_schema)
        if sign:
            mappings, char_to_id, id_to_char, tag_to_id, id_to_tag = self.loader.char_mapping(train_sentence,
                                                                                              self.config.lower, sign)
            train_data = self.loader.prepare_dataset(train_sentence, char_to_id, tag_to_id, self.config.lower)
        else:
            train_data = self.loader.prepare_dataset(train_sentence, char_to_id, tag_to_id, self.config.lower)

        print('train 预处理后数据长度：{}'.format(len(train_data)))

        batch_size = self.config.batch_size if sign else 100
        batch_data = self.loader.batch_size_padding(train_data, batch_size)
        return batch_data, id_to_tag, tag_to_id, char_to_id

    def evaluate(self, sess, trans, data_manager, id_to_tag):
        ner_results = self._evaluate(sess, trans, data_manager, id_to_tag)
        report = self.util.report_ner(ner_results, self.config.report_file)
        return report

    def _evaluate(self, sess, trans, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = trans.eval()  # tensor.eval() 相当于 sess.run(self.trans)作用；其实就是执行
        for batch in data_manager:
            strings, chars, _, tags = batch
            lengths, scores = self._run_sess(sess, batch, False)
            batch_paths = self._decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = self.loader.iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = self.loader.iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results



    def _decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.config.num_tags + [0]])  # 初始化一个
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])  # 创建一个字符长度是 输入字长度维度元素为1的np数组
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path[1:])
        return paths



    def _run_sess(self, sess, batch, is_train):
        self._create_feed_dict(batch)
        if is_train:
            loss, train_op, lengths, trans, global_step, learn_rate = sess.run(
                [self.model.loss, self.model.train_op, self.model.lengths, self.model.trans, self.model.global_step,
                 self.model.lr], self.feed_dict)
            return loss, lengths, trans, global_step, learn_rate
        else:
            lengths, logits = sess.run([self.model.lengths, self.model.logits], self.feed_dict)
            return lengths, logits



    def _create_feed_dict(self, batch, is_train=True):
        _, chars, segs, tags = batch
        self.feed_dict = {
            self.model.char_inputs: np.asarray(chars),
            self.model.seg_inputs: np.asarray(segs),
            self.model.dropout: 1.0,
        }
        if is_train:
            self.feed_dict[self.model.targets] = np.asarray(tags)
            self.feed_dict[self.model.dropout] = self.config.dropout


    def _evaluate_line(self, sess, inputs, id_to_tag):
        '''
        :param sess:
        :param inputs:
        :param id_to_tag:
        :return:
        '''
        trans = self.model.trans.eval(session=sess)
        lengths, scores = self._run_sess(sess, inputs, False)
        batch_paths = self._decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return self.util.result_to_json(inputs[0][0], tags)


    def train(self):

        batch_data, id_to_tag, tag_to_id, char_to_id = self._data_preprocess(self.config.train_file, self.config.zeros,
                                                                             True)
        self.logger.info('train data prepare done')
        dev_batch_data, _, _, _ = self._data_preprocess(self.config.dev_file, self.config.zeros, False, char_to_id,
                                                        tag_to_id)
        self.logger.info('dev data prepare done')

        self.logger.info('start train......')
        batch_len = len(batch_data)

        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True  这个是动态允许使用gpu空间
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.config.max_epoch):
                lr = ''
                ls = ''
                random.shuffle(batch_data)
                for step in range(batch_len):
                    loss, lengths, trans, global_step, learn_rate = self._run_sess(sess, batch_data[step], True)
                    if step == (batch_len - 1):
                        lr, ls = learn_rate, loss
                    if (int(step) + 1) % self.config.steps_check == 0:
                        self.logger.info(
                            ' epoch:{}, step/total_batch:{}/{}, global_step:{}, learn_rate:{}, loss:{}'.format(epoch,
                                                                                                               step,
                                                                                                               batch_len,
                                                                                                               global_step,
                                                                                                               learn_rate,
                                                                                                               loss))
                if (epoch + 1) % 2 == 0:
                    print('*' * 50)
                    report = self.evaluate(sess, self.model.trans, dev_batch_data, id_to_tag)
                    self.logger.info(report[1].strip())
                    self.logger.info('dev: epoch:{},  learn_rate:{}, loss:{}'.format(epoch, lr, ls))
                    if (int(epoch) + 1) % 20 == 0:
                        self.save_model(sess, epoch)



    def online(self, inputs):
        if not inputs:
            return json.dumps({'result':'error'})

        with open('./data/id_to_tag.txt', 'r', encoding='utf-8') as tag, open('./data/char_to_id.txt', 'r',
                                                                              encoding='utf-8') as char:
            id_to_tag = {int(line.strip().split(':')[0]): line.strip().split(":")[1] for line in tag.readlines()}
            char_to_id = {s[0:s.rfind(':')].strip(): int(s[s.rfind(':') + 1:].strip()) for s in char.readlines()}

        self.saver = tf.train.Saver()
        tf_config = tf.ConfigProto()             # 实例化一个设置GPU的对象  函数用在创建session的时候，用来对session进行参数配置
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=tf_config) as sess:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.logger.info('restore model')
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                data = self.loader.input_from_line(inputs, char_to_id)
                result = self._evaluate_line(sess, data, id_to_tag)
                return json.dumps(result)



# if __name__ == '__main__':
#     main = main('online')


