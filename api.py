# -*- coding: utf-8 -*-
# @Time   : 2018/11/30 10:31
# @Author : Richer
# @File   : api.py
# 服务类api接口

from flask import Flask,request
from run import Run
import json

app = Flask(__name__)
@app.route('/', methods=['GET','POST'])
def get_result():
    if not request.args or 'input' not in request.args:
        # 没有指定imgage则返回全部
        return json.dumps({'result':'args error'})
    else:
        try:
            input = request.args['input']
            main  = Run(type = 'line')
            res = main.online(input)
            return res
        except:
            return json.dumps({'result': 'error'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)