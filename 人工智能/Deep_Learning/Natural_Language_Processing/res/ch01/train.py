# coding: utf-8
import sys
sys.path.append('..')  # 为了引入父目录的文件而进行的设定
from 人工智能.Deep_Learning.Natural_Language_Processing.res.common.optimizer import SGD
from 人工智能.Deep_Learning.Natural_Language_Processing.res.common.trainer import Trainer
from 人工智能.Deep_Learning.Natural_Language_Processing.res.dataset import spiral
from 人工智能.Deep_Learning.Natural_Language_Processing.res.ch01.two_layer_net import TwoLayerNet


# 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()
