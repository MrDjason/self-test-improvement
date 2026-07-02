# ==================== 导入模块 ====================
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# d2l.use_svg_display()
# ==================== 3.5.1 读取数据集 ====================
# 下载并预处理 Fashion-MNIST 数据集，使其能被 PyTorch 模型直接使用
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True)
print(len(mnist_train), len(mnist_test),sep='\n')
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):
    '''返回Fashion-MNIST数据集的文本标签'''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    '''绘制图像列表'''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
batch_size = 18
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
X, y = next(iter(train_iter)) 
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.show()

# ==================== 3.5.2 读取小批量 ====================
batch_size = 256
def get_dataloader_workers():
    '''使用4个进程来读取数据'''
    return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop:.2f} sec')

# ==================== 3.5.3 整合所有组件 ====================
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root='../data', train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root='../data', train=False, transform=trans, download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shade, y.dtype)
    break