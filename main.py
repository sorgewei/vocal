import os
import torch
import torchaudio
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import librosa.display
from torchaudio.transforms import MelSpectrogram

data_dir = 'test2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集
class Dataset (torch.utils.data.Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir = data_dir
        self.transform_dir = transform
        self.audio_files = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if i.endswith('.wav')]

    def __len__(self):
        return len(self.audio_files)

# 模型神经网络
class Module1 (nn.Module):
    def __init__(self):
        pass


    def forward(self):
        pass

# 预处理，梅尔音频转化

# TensorBoard
writer = SummaryWriter(log_dir='./logs')

# 优化器，损失函数

# 训练
epochs = 10
for epoch in range(epochs):
    pass

