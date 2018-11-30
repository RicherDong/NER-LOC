import tensorflow as tf
import sys,os

class Config:

      seg_dim  = 20      #切词信息维度
      char_dim = 100     #字向量模型维度
      lstm_dim = 100     #lstm 内部维度
      dropout  = 0.5
      learn_rate = 0.001  #学习率
      max_epoch  = 400    #最大训练次数
      batch_size = 64
      steps_check = 300   # 检查频率
      num_tags    = 51
      num_chars   = 2641
      num_segs    = 4     # 切词信息 四维   i b o e
      filter_width = 3    # 卷积核大小
      repeat_times = 4    # 膨胀卷积时卷积次数


      clip = 5
      optimizer = 'adam'
      model_type = 'idcnn' # 训练模型
      tag_schema = 'iobes'
      pre_emb = True
      lower   = False
      zeros   = True
      clean   = True

      root_path = os.getcwd() + os.sep
      # ckpt_path = os.path.join(root_path + 'ckpt', "")          # 模型路径
      cnn_ckpt_path = os.path.join(root_path + 'ckpt\idcnn', '')
      lstm_ckpt_path = os.path.join(root_path + 'ckpt\lstm', '')
      log_file  = os.path.join(root_path + 'log', 'train.log')     # 训练日志记录
      train_file = os.path.join(root_path + 'data', 'example.train') # 训练数据集
      dev_file  = os.path.join(root_path + 'data', 'example.dev')  # 验证数据集
      test_file = os.path.join(root_path + 'data', 'example.test') # 测试数据集
      report_file= os.path.join(root_path + 'result', 'ner_predict.utf8')  # 测试数据集

      assert  0 < dropout< 1, 'dropout must between 0, 1'
      assert  learn_rate > 0, 'learn_rate must > 0'
      assert  optimizer in ['adam', 'sgd', 'adagrad'] , 'this optimizer not exist'
