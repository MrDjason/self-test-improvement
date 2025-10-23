# ==================== 导入模块 ====================
import numpy
import matplotlib.pyplot
# ==================== 定 义 类 ====================
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
    def train():
        pass
    def query(): 
        pass
# ==================== 主程序 ====================
inputnodes, hiddennodes, outputnodes, learningrate = 3,3,3,0.3

neural_network_1 = NeuralNetwork(inputnodes,hiddennodes,outputnodes,learningrate)
# numpy.random.rand(rows, columns) 生成 3X3 Numpy数组，数组中每个值为0-1
numpy.random.rand(3, 3) - 0.5 # 生成初始的且数值在[-0.5,0.5]的权重矩阵
