import os
import torch
import torchaudio
from sympy.physics.units import length
from torch import nn
import torch.nn as nn
from torch._export.db.examples.assume_constant_result import model
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter, writer
import librosa.display
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = {
    "data_dir": "text2 - 副本",       # 数据集路径
    "sample_rate": 16000,         # 音频采样率
    "n_mels": 128,                # Mel频带数
    "n_fft": 1024,                # FFT窗口大小（决定频谱图宽度）
    "hop_length": 512,            # FFT跳长
    "num_classes": 10,            # 类别数（根据实际修改）
    "batch_size": 32,
    "lr": 1e-3,                   #训练步长
    "epochs": 20
}


# 数据集
class AudioDataset (torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.transform_dir = transform

        self.audio_files = []
        self.labels = []

        # 取音频
        labels = []
        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                try:
                    self.audio_files.append(os.path.join(data_dir, file))
                    labels_str = file.split('-')[0]
                    # labels.append(os.path.join(data_dir, file))
                    # 把label变成整数，不知道为什么但是不这样会报错
                    labels = int(labels_str)

                    self.labels.append(labels)
                except (IndexError, ValueError) as e:
                    print(f"跳过无效文件 {file}，错误原因：{str(e)}")

# dataset = AudioDataset(config["data_dir"])
# print(f"Dataset size: {len(dataset)}")
# print(f"First item: {dataset[0]}")


    def __len__(self):
        return len(self.audio_files)
    # 取文件
    def __getitem__(self, idx):

        waveform, sr = torchaudio.load(self.audio_files[idx])
        waveform = torchaudio.transforms.Resample(sr, config["sample_rate"])(waveform)

        # 音频长度
        target_length = config["sample_rate"] * 4
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_length]

        # 数据增强：随机时移 不会
        shift = torch.randint(0, target_length, (1,)).item()
        waveform = torch.roll(waveform, shifts=shift, dims=1)

        # 转换为Mel频谱
        if self.transform:
            mel_spec = self.transform(waveform)
        else:
            mel_spec = waveform  # 备用


        return mel_spec, torch.tensor(self.labels[idx])
        # audio_path = self.audio_files[idx]
        # waveform, sample_rate = torchaudio.load(audio_path)
        # if self.transform:
        #     waveform = self.transform(waveform)
        # #     参数暂时设为0
        # return waveform, torch.tensor(0)


    # 模型神经网络
class Module1 (nn.Module):
    # num_classes待修改
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((6, 6))
        )  # 自适应池化解决尺寸问题

        self.classifier = nn.Sequential(
            nn.Linear(64 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 预处理，梅尔音频转化
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=config["sample_rate"],
    n_mels=config["n_mels"],
    n_fft=config["n_fft"],
    hop_length=config["hop_length"]
)

# 加载数据集并划分训练集/验证集
dataset = AudioDataset(config["data_dir"], transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


# 优化器，损失函数
model = Module1(config["num_classes"]).to(device)
loss_fn = nn.CrossEntropyLoss()

model = Module1().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

writer = SummaryWriter()

best_val_acc = 0.0

# 训练

train_step = 0
start_time = time.time()

total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数

for epoch in range(config["epochs"]):

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    print("----------第{}轮-----------".format(epoch))

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU

        # 添加通道维度 (batch, 1, n_mels, time)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_step += 1

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}, loss：{}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)

#   测试步骤
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():  # 无梯度，不进行调优

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_acc)

        # tensorboard
        writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_epoch{epoch}.pth")

        torch.save(Module1, "Module1_{}_gpu.pth")  # 保存
        print("模型已保存")

writer.close()



# for inputs, labels in train_loader:
#     # voc, labels = data
#     voc = voc.to(device)
#     labels = labels.to(device)
#     outputs = Module1(voc)
#     loss = loss_fn(outputs, labels)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
#     # 求整体测试数据集上的误差或正确率
#     total_test_loss = total_test_loss + loss.item()
#     accuracy = (outputs.argmax(1) == labels).sum()
#     total_accuracy = total_accuracy + accuracy
# print("整体测试集上的Loss：{}".format(total_test_loss))
# print("整体测试集上的正确率：{}".format(total_accuracy / length))  # 此处长度未赋值
# writer.add_scalar("test_loss", total_test_loss, total_test_step)
# writer.add_scalar("test_accuracy", total_test_loss, total_test_step, total_test_step)
# total_test_step += 1
