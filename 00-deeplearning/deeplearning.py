import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader

#配置训练环境和超参数
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

batch_size = 256
num_workers = 0
lr = 1e-4
epochs = 23
#  经debug，epoch在20-25轮左右，拟合效果达最佳，accuracy约92%且基本不变

#  数据变换设置
from torchvision import transforms

image_sizes = 28
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_sizes),
    transforms.ToTensor()
])

#  读取数据方式
class FMDataset(Dataset):
    def __init__(self, df , transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:,0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else :
            image = torch.tensor(image/255 ,dtype= torch.float)
        label = torch.tensor(label,dtype = torch.long)
        return image, label

#导入训练集和测试集
train_df = pd.read_csv("../data/FashionMNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("../data/FashionMNIST/fashion-mnist_test.csv")
train_data = FMDataset(train_df, data_transforms)
test_data = FMDataset(test_df, data_transforms)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# 结果可视化
import matplotlib.pyplot as plt
image, label = next(iter(train_loader))
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")


# 模型构建
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x

#  选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#  把模型搬到设备上
model = Net()
model = model.to(device)

#  设定损失函数
criterion = nn.CrossEntropyLoss()

#  设定优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)


#   训练流程
def train(epoch):
    model.train()
    running_loss =0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        #  梯度清零
        optimizer.zero_grad()
        #  前向传播
        output =model(data)
        #  计算损失
        loss = criterion(output, target)
        #  反向传播
        loss.backward()
        #  更新参数
        optimizer.step()
        #  统计loss
        running_loss += loss.item()
        #  打印进度
        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
            )

#   测试流程
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    #  不需要初始化优化器和更新优化器，因为验证过程不需要“学习”
    #  不需要计算梯度，节省显存
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #  前向传播
            output = model(data)
            #  计算batch loss
            test_loss += criterion(output, target).item()
            #  计算分类准确率，找到概率最大的类别的索引
            predict = output.argmax(dim=1, keepdim=True)
            #  对比预测值和真实值，统计预测正确的数量
            correct += predict.eq(target.view_as(predict)).sum().item()
    #  计算平均loss
    test_loss /=len(test_loader)
    #  打印测试结果
    print(f'\nTest result: Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{len(test_loader.dataset)} '
        f'({100. * correct / len(test_loader.dataset):.2f}%)\n')



if __name__ == '__main__':
    print(f"开始训练，共 {epochs} 轮...")
    plt.show()

    for epoch in range(1, epochs + 1 ):
        train(epoch)
        test(epoch)

    print("finish!")

    save_path ="./FashionModel.pkl"
    torch.save(model, save_path)
    print(f"已保存至: {save_path}")

