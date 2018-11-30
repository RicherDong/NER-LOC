# -*- coding: utf-8 -*-
# @Time   : 2018/11/12 15:46
# @Author : Richer
# @File   : util.py
'''工具类'''
import logging
import sys,os
from conlleval import return_report
import shutil

class Util():

    def __init__(self):
        pass


    def make_path(self, params):
        if not os.path.isdir(params.result_file):
            os.makedirs(params.result_file)
        if not os.path.isdir(params.ckpt_path):
            os.makedirs(params.ckpt_path)
        if not os.path.isdir(params.log):
            os.makedirs(params.log)


    def get_logger(self, log_file):  # train.log
        # 1、创建一个logger
        logger = logging.getLogger(log_file)  # <Logger train.log (WARNING)>
        logger.setLevel(logging.DEBUG)  # 设置训练时的日志记录级别为debug级别
        # 2、创建一个handler，用于写入日志文件
        fh = logging.FileHandler(log_file)  # 用来写入日志的文件
        fh.setLevel(logging.DEBUG)
        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 3、定义handler的输出格式（formatter）
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # 4、给handler添加formatter
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # 5、给logger添加handler
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger


    def clean(self, config):
        """
        Clean current folder
        remove saved model and training log
        """
        print(config.log_file)
        char2id_file = './data/char_to_id.txt'
        id2char_file = './data/id_to_char.txt'
        tag2id_file = './data/tag_to_id.txt'
        id2tag_file = './data/id_to_tag.txt'
        # if os.path.isfile(config.log_file):
        #     os.remove(config.log_file)

        if os.path.isfile(config.report_file):
            os.remove(config.report_file)

        if os.path.isdir("__pycache__"):
            shutil.rmtree("__pycache__")

        if os.path.isfile(char2id_file):
            os.remove(char2id_file)

        if os.path.isfile(id2char_file):
            os.remove(id2char_file)

        if os.path.isfile(tag2id_file):
            os.remove(tag2id_file)

        if os.path.isfile(id2tag_file):
            os.remove(id2tag_file)


    def report_ner(self, results, output_file):
        """
        Run perl script to evaluate model
        """
        with open(output_file, "w", encoding='utf8') as f:
            to_write = []
            for block in results:
                for line in block:
                    to_write.append(line + "\n")
                to_write.append("\n")

            f.writelines(to_write)
        eval_lines = return_report(output_file)
        return eval_lines


    def result_to_json(self, string, tags):
        item = {"string": string, "entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
        return item