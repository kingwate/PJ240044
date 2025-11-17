from torch.utils.data import DataLoader
import random
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.nn.parameter import Parameter
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


class ContrastiveTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),  # 随机裁剪并缩放到224x224
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)  # 生成两个不同的增强视图


image_path = ("../数据集/甲骨文多模态数据集")
pre_train_dataset = datasets.ImageFolder(
    root=os.path.join(image_path, "train"),
    transform=ContrastiveTransform()  # 返回tuple (view1, view2)
)


def collate_fn(batch):
    # 将[(view1, view2, label), ...] 转换为 ([view1s, view2s], labels)
    view1 = torch.stack([x[0][0] for x in batch])
    view2 = torch.stack([x[0][1] for x in batch])
    labels = torch.tensor([x[1] for x in batch])
    return torch.cat([view1, view2], dim=0), labels


pre_train_loader = DataLoader(
    pre_train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
)


# 损失函数定义
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z):
        batch_size = z.shape[0] // 2
        device = z.device

        # 生成标签：正样本是对应的另一个增强视图
        labels = torch.cat([
            torch.arange(batch_size, device=device) + batch_size,
            torch.arange(batch_size, device=device)
        ])

        # 计算余弦相似度矩阵 (2N, 2N)
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # 易错点：排除自身样本的相似度（将对角线设为负无穷）
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim_matrix.masked_fill_(mask, -float('inf'))

        # 计算交叉熵损失
        loss = self.cross_entropy(sim_matrix, labels)
        return loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, labels):
        batch_size = z.shape[0] // 2
        device = z.device

        # 创建标签矩阵
        labels_expand = labels.repeat(2)
        mask = torch.eq(labels_expand.unsqueeze(0), labels_expand.unsqueeze(1)).float()
        mask = mask - torch.eye(2 * batch_size, device=device)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # 数值稳定性改进：Log-Sum-Exp 技巧
        max_sim = sim_matrix.max(dim=1, keepdim=True).values.detach()  # 每行最大值
        sim_stable = sim_matrix - max_sim  # 减去最大值防止指数爆炸
        exp_sim = torch.exp(sim_stable)

        # 计算对数概率
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True)) + max_sim  # 补偿最大值
        log_prob = sim_matrix - log_sum_exp

        # 计算损失
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = -mean_log_prob.mean()

        return loss

# 模型定义
device = torch.device("cuda:0")
feature_length = 128
net = models.resnet18()
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, feature_length)
net.to(device)
# 后续可能调整为64
loss_function = InfoNCELoss(temperature=0.07)
# 预训练轮数
epochs = 50

# print(fc_in_features) 为512

optimizer = optim.Adam(net.parameters(), lr=0.0005)
pre_train_steps = len(pre_train_loader)
for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    pre_train_bar = tqdm(pre_train_loader, file=sys.stdout)
    for step, data in enumerate(pre_train_bar):
        # 标签不重要
        images, labels = data
        optimizer.zero_grad()
        images = images.to(device)

        outputs = net(images)
        loss = loss_function(outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pre_train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1, epochs, loss)

    print('[epoch %d] train_loss: %.4f  ' % (epoch + 1, running_loss / pre_train_steps))


#################################
#第二阶段预训练还是

#################################



# 加载无标记完整甲骨数据集
unlabeled_image_path = "../数据集/无标记完整甲骨"
unlabeled_dataset = datasets.ImageFolder(
    root=unlabeled_image_path,
    transform=ContrastiveTransform()
)

unlabeled_loader = DataLoader(
    unlabeled_dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn,
)

# 第二阶段：监督对比学习预训练
print("开始第二阶段：监督对比学习预训练")
epochs = 50
net.fc = nn.Linear(fc_in_features, feature_length)  # 重置为特征提取器
net.to(device)
loss_function = SupervisedContrastiveLoss(temperature=0.07)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(unlabeled_loader, file=sys.stdout)
    
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_bar.desc = "supervised contrastive train epoch[{}/{}] loss:{:.4f}".format(
            epoch + 1, epochs, loss)
    
    print('[epoch %d] supervised_contrastive_loss: %.4f' % (epoch + 1, running_loss / len(unlabeled_loader)))

# 保存监督对比学习预训练模型
torch.save(net.state_dict(), './ResNet18_supervised_contrastive.pth')



#################################
#微调过程开始

#################################




# 正式开始训练过程，首先进行训练集和测试集的加载
data_transform = {
    "train": transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),  # 图片转换成形状为(C, H, W)的Tensor格式
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "test": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

image_path = "../数据集/训练测试19_25.3.1"
batch_size = 16

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, pin_memory=True)

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=data_transform["test"])

validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=32,
                                              shuffle=False, pin_memory=True)

# print(fc_in_features) 为512
net.fc = nn.Linear(in_features=512, out_features=2).to(device)
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.00005)

epochs = 50
best_acc = 0.0
best_recall = 0.0
best_kappa = 0.0
best_f1 = 0.0
best_precision = 0.0

train_steps = len(train_loader)
for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1, epochs, loss)

        # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    all_predict_y = []
    all_val_labels = []
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            outputs = net(val_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels).sum().item()
            all_predict_y.extend(predict_y.cpu().numpy())
            all_val_labels.extend(val_labels.cpu().numpy())

    val_num = len(validate_loader.dataset)
    val_accurate = acc / val_num
    val_predict_y = np.array(all_predict_y)
    val_true_y = np.array(all_val_labels)

    # Calculate evaluation metrics
    val_recall = recall_score(val_true_y, val_predict_y, average='macro')
    val_f1 = f1_score(val_true_y, val_predict_y, average='macro')
    val_precision = precision_score(val_true_y, val_predict_y, average='macro')
    val_kappa = cohen_kappa_score(val_true_y, val_predict_y)

    print(
        '[epoch %d] train_loss: %.4f  val_accuracy: %.4f  val_recall: %.4f  val_f1: %.4f  val_precision: %.4f  val_kappa: %.4f' %
        (epoch + 1, running_loss / train_steps, val_accurate, val_recall, val_f1, val_precision, val_kappa))

    save_path = f'./ResNet18_fold{epoch}.pth'
    best_acc = val_accurate
    best_recall = val_recall
    best_kappa = val_kappa
    best_f1 = val_f1
    best_precision = val_precision
    torch.save(net.state_dict(), save_path)

# Print the best metrics values for the fold
print(f'best_acc: {best_acc:.4f}%')
print(f'best_Recall: {best_recall:.4f}%')
print(f'best_F1-score: {best_f1:.4f}%')
print(f'best_Precision: {best_precision:.4f}%')
print(f'best_Kappa: {best_kappa:.4f}%')
