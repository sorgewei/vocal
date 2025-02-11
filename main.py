import os
import torch
import torchaudio
from torch import nn
import torch.nn as nn
from torch._export.db.examples.assume_constant_result import model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import librosa.display
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集
class Dataset (torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = None
        self.data_dir = data_dir
        self.transform_dir = transform
        # 取音频
        self.audio_files = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if i.endswith('.wav')]

    def __len__(self):
        return len(self.audio_files)
    # 取文件
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)
        #     参数暂时设为0
        return waveform, torch.tensor(0)


    # 模型神经网络
class Module1 (nn.Module):
    def __init__(self):
        super(Module1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # 假设 MelSpectrogram 输出尺寸为 128x128
        self.fc2 = nn.Linear(128, 10)  # 假设有10个类别

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 预处理，梅尔音频转化
transform = MelSpectrogram(sample_rate=16000, n_mels=128)
train_dataset = Dataset(data_dir='test2', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# TensorBoard
writer = SummaryWriter(log_dir='./logs')

# 优化器，损失函数
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
model = Module1().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


# 训练

epochs = 10
train_step = 0
start_time = time.time()
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    print("----------第{}轮-----------".format(epoch))

    for i, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        train_step += 1

        if train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}, loss：{}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)

#     测试步骤没写



writer.close()


