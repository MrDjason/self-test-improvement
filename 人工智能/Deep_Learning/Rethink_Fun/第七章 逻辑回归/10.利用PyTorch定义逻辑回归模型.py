# 利用PyTorch定义一个逻辑回归模型
# nn.Module是PyTorch中核心组件，复杂模型和内部的各个可训练模块，都必须继承自nn.Module
'''
nn.Module带来了以下好处：
模块化构建：对于复杂的网络，你可以将它划分为多个子模块进行构建。然后将它们组合为一个复杂模型。
自动管理参数：nn.Module会自动追踪模块内所有的参数，无论这些参数是嵌套在子模块中还是作为属性值直接存储在当前模块中。可以通过模块的parameters()或者named_parameters()进行查看。
统一的forward方法：所有继承自nn.Module的类必须实现一个forward方法，这样各个模块之间根据组合关系对数据进行前向处理。PyTorch里的计算图和自动求梯度机制会帮助我们实现反向传播。
统一的设备管理：利用nn.Module提供的.to(device)方法，可以方便的将整个模型迁移到GPU或者CPU。
模型的保存和加载：nn.Module提供了标准的模型保存和加载方法。
定义了模型的train和eval状态：有的模块在训练时和预测（或者叫推理）时前向传播实现是不同的，可以通过model.train()或者model.eval()统一切换本身，和内部包含的所有模块的状态。
'''

# ==================== 导入模块 ====================
import torch
import torch.nn as nn 
# ==================== 7.10.2 实现逻辑回归模型 ====================
# y=sigmoid(wx+b)的代码实现

# 定义一个模型需要继承nn.Moudle，必须实现：__init__、forward

class LogisticRegressionModel(nn.Module): # 定义一个类继承自nn.Module
    def __init__(self, input_dim): # 类的构造函数，创建模型实例自动调用 
    # input_dim 输入特征维度
        super().__init__() # 调用父类nn.Module的初始化方法，完成父类的初始化工作
        self.linear = nn.Linear(input_dim, 1) # nn.Linear也继承来自nn.Module
        # 输入为input_dim，输出为1个值
        # 定义模型的核心组件，线性层nn.Linear并作为模型的属性（self.linear）
    def forward(self, x): # 定义模型的前向传播逻辑，即数据从输入到输出的计算过程
        return torch.sigmoid(self.linear(x)) # Logistic Regression 输出概率