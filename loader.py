# -*- coding: utf-8 -*-
# @Time   : 2018/11/12 15:46
# @Author : Richer
# @File   : loader.py

import tensorflow as tf
import collections
import numpy as np
import math
import jieba
import re
import sys,os

class Loader:
    def __init__(self):
       pass


    def load_sentences(self, path, zeros):

        sentence, sentences = [],[]  # 第一个是临时存储， 第二个存放整个数据集。多个句子的组合

        def _zero_digits(str):
            return re.sub('\d', '0', str)

        file = open(path, 'r', encoding='utf-8')
        lines = file.readlines()
        for  line in lines:
             line = _zero_digits(line.rstrip()) if zeros else line.rstrip()  # 数字 0 化开启, 就将行里面数字变为 0 ;
             if not line:
                 if len(sentence) > 0:       # 此位置代表的是换行的时候
                     sentences.append(sentence)
                     sentence = []
             else:
                 if  line[0] == " ":
                     line = "$" + line[1:]
                 word = line.split()
                 sentence.append(word)
        if len(sentence) > 0:
            sentences.append(sentence)
        return sentences


    def update_tag_schema(self, sentences, tag_scheme):
        '''
        :param sentences:
        :param tag_scheme:
        :return:
        '''
        for i, s in enumerate(sentences):
            tags = [w[-1] for w in s]
            if not self._iob(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                raise Exception('Sentences should be given in IOB format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))
            if tag_scheme == 'iob':
                for word, new_tag in zip(s, tags):
                    word[-1] = new_tag
            elif tag_scheme == 'iobes':
                new_tags = self._iob_iobes(tags)
                for word, new_tag in zip(s, new_tags):
                    word[-1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')


    def char_mapping(self, sentences, lower, sign= False):
        chars = []
        tags = []
        for s in sentences:
            for x in s:
                chars.append(x[0].lower() if lower else x[0])
                tags.append(x[-1])
        chars_list = collections.Counter(chars)   # 词数统计
        tags_list = collections.Counter(tags)     # tag统计
        print('char_list总长度：{}'.format(len(chars_list)))
        print('tag_list 总长度：{}'.format(len(tags_list)))
        char_max = chars_list.most_common()
        tags_max = tags_list.most_common()

        # char信息处理
        completion =  [('<PAD>', 10000001),('<UNK>', 10000000)]
        char_max = completion + char_max
        id_to_char = {i: v[0] for i, v in  enumerate(char_max)}    # {0: '<PAD>', 1: '<UNK>', 2: '0', 3: '，', 4: '：', 5: '。', 6: '无',}
        char_to_id = dict(zip(id_to_char.values(), id_to_char.keys()))
        # tag信息
        id_to_tag = {i: v[0] for i, v in enumerate(tags_max)}
        tag_to_id = dict(zip(id_to_tag.values(), id_to_tag.keys()))
        if sign:
            with open('./data/id_to_tag.txt', 'w', encoding='utf-8') as fid2tag, \
                 open('./data/tag_to_id.txt', 'w', encoding='utf-8') as ftag2id, \
                 open('./data/id_to_char.txt', 'w', encoding='utf-8') as fid2char,\
                 open('./data/char_to_id.txt', 'w', encoding='utf-8') as fchar2id:

                for k, v in id_to_tag.items():
                    fid2tag.write(str(k) + ":" + str(v) + "\n")
                for k, v in tag_to_id.items():
                    ftag2id.write(k + ":" + str(v) + "\n")
                for k, v in id_to_char.items():
                    fid2char.write(str(k) + ":" + str(v) + "\n")
                for k, v in char_to_id.items():
                    fchar2id.write(k + ':' + str(v) + "\n")

        return char_max, char_to_id, id_to_char, tag_to_id, id_to_tag


    def prepare_dataset(self, sentences, char_to_id, tag_to_id, lower=False, train=True):
        '''
        整理数据
        :param sentences:
        :param char_to_id:
        :param tag_to_id:
        :param lower:
        :param train:
        :return:
        '''
        none_index = tag_to_id["O"]

        def f(x):
            return x.lower() if lower else x

        data = []
        for s in sentences:
            string = [w[0] for w in s]
            chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']  for w in string]
            segs = self._get_seg_features("".join(string))  # 句子按sbie分词标注
            if train:
                tags = [tag_to_id[w[-1]] for w in s]  # 标注命名实体分词id
            else:
                tags = [none_index for _ in chars]
            data.append([string, chars, segs, tags])
        return data



    # def prepare_dataset2(self, sentences, char_to_id, tag_to_id, lower=False, train=True):
    #     '''
    #     整理数据
    #     :param sentences:
    #     :param char_to_id:
    #     :param tag_to_id:
    #     :param lower:
    #     :param train:
    #     :return:
    #     '''
    #     none_index = tag_to_id["O"]
    #
    #     def f(x):
    #         return x.lower() if lower else x
    #
    #     data = []
    #     for s in sentences:
    #         for str in s:
    #             # string = [w[0] for w in s]
    #             # chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']  for w in string]
    #             # segs = self._get_seg_features("".join(string))  # 句子按sbie分词标注
    #             # if train:
    #             #     tags = [tag_to_id[w[-1]] for w in s]  # 标注命名实体分词id
    #             # else:
    #             #     tags = [none_index for _ in chars]
    #             # data.append([string, chars, segs, tags])
    #     return data




    def batch_size_padding(self, sentences, batch_size):
        # sentences 是所有的的数据
        def data_padding(data):
            strings, chars, segs, targets = [], [], [], []     # 一句话, 字段
            strings_append, chars_append, segs_append, target_append = strings.append, chars.append, segs.append, targets.append
            data_len = max([len(s[0]) for s in data])
            for str in data:
                string, char, seg, target = str
                padding_zero = [0] * (data_len - len(string))
                strings_append(string + padding_zero)
                chars_append(char + padding_zero)
                segs_append(seg + padding_zero)
                target_append(target + padding_zero)
            return [strings, chars, segs, targets]


        data_sort = sorted(sentences, key= lambda x: len(x[0]))    #按行进行排序
        num_batch = int(math.ceil(len(sentences)/batch_size))              #一次训练里面需要遍历的次数
        batch_list = list()

        for i in range(num_batch):
            if len(data_sort[i * int(batch_size)][0]) < 5:     #一个训练循环里面数量小于5的跳过
                continue
            data = data_padding(data_sort[i * int(batch_size) : (i+1) * int(batch_size)])
            batch_list.append(data)
        return batch_list


    def _get_seg_features(self, string):
        """
        Segment text with jieba
        features are represented in bies format
        s donates single word
        """
        seg_feature = []
        for word in jieba.cut(string):
            if len(word) == 1:
                seg_feature.append(0)
            else:
                tmp = [2] * len(word)
                tmp[0] = 1
                tmp[-1] = 3
                seg_feature.extend(tmp)
        return seg_feature


    def input_from_line(self, line, char_to_id):
        """
        Take sentence data and return an input for
        the training or the evaluation function.
        """
        line = self._full_to_half(line)  # 去除空格, 特殊的符号
        line = self._replace_html(line)  # 将实体标签变更为
        inputs = list()
        inputs.append([line])
        line.replace(" ", "$")
        inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                        for char in line]])
        inputs.append([self._get_seg_features(line)])           # inputs[0] 代表的是处理后的汉字, inputs[1] 是对应的词位置， inputs[2] 代表的是对应的tag
        inputs.append([[]])
        return inputs


    def _full_to_half(self, s):
        """
        Convert full-width character to half-width one
        """
        n = []
        for char in s:
            num = ord(char)
            if num == 0x3000:  # 空格
                num = 32
            elif 0xFF01 <= num <= 0xFF5E:  # 特殊符号
                num -= 0xfee0
            char = chr(num)
            n.append(char)
        return ''.join(n)


    def _replace_html(self, s):
        s = s.replace('&quot;', '"')
        s = s.replace('&amp;', '&')
        s = s.replace('&lt;', '<')
        s = s.replace('&gt;', '>')
        s = s.replace('&nbsp;', ' ')
        s = s.replace("&ldquo;", "")
        s = s.replace("&rdquo;", "")
        s = s.replace("&mdash;", "")
        s = s.replace("\xa0", " ")
        return (s)


    def _iob(self, tags):
        """
        Check that tags have a valid IOB format.
        Tags in IOB1 format are converted to IOB2.
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            split = tag.split('-')
            if len(split) != 2 or split[0] not in ['I', 'B']:
                return False
            if split[0] == 'B':
                continue
            elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
            elif tags[i - 1][1:] == tag[1:]:
                continue
            else:
                tags[i] = 'B' + tag[1:]
        return True


    def _iob_iobes(self, tags):
        """
        IOB -> IOBES
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag == 'O':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'B':  # 如果开头是 B
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':  # 如果下一个不是最后一个， 下一个是中间的；
                    new_tags.append(tag)  # 当下tag 放到新的tags里面
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif tag.split('-')[0] == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise Exception('Invalid IOB format!')
        return new_tags


    def iobes_iob(self, tags):
        """
        IOBES -> IOB
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag.split('-')[0] == 'B':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'I':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'S':
                new_tags.append(tag.replace('S-', 'B-'))
            elif tag.split('-')[0] == 'E':
                new_tags.append(tag.replace('E-', 'I-'))
            elif tag.split('-')[0] == 'O':
                new_tags.append(tag)
            else:
                raise Exception('Invalid format!')
        return new_tags