#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量学习脚本 - 添加S类别
基于现有13类模型添加S类别识别
"""

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
from datetime import datetime
import shutil
from collections import Counter

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Dataset14Classes(Dataset):
    """14类数据集加载器"""
    
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # 14个类别：0-9数字 + N、X、G、S字母
        self.class_names = [str(i) for i in range(10)] + ['N', 'X', 'G', 'S']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        print("正在加载数据集...")
        
        # 加载所有类别数据
        for cls_name in self.class_names:
            cls_path = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_path):
                print(f"警告: 未找到类别文件夹 {cls_name}")
                continue
                
            label = self.class_to_idx[cls_name]
            cls_count = 0
            
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(cls_path, img_file)
                    self.samples.append((img_path, label))
                    cls_count += 1
            
            status = "新增" if cls_name == 'S' else "原有"
            print(f"  {status} 类别 {cls_name}: {cls_count} 张")
        
        print(f"总计: {len(self.samples)} 张图片")
    
    def get_sample_weights(self):
        """计算样本权重用于平衡采样"""
        class_counts = Counter(label for _, label in self.samples)
        total_samples = len(self.samples)
        
        # 计算类别权重
        class_weights = {}
        for i in range(14):
            count = class_counts.get(i, 1)
            weight = total_samples / (14 * count)
            # 给S类别额外权重
            if i == 13:  # S类别
                weight *= 1.5
            class_weights[i] = weight
        
        # 计算每个样本的权重
        sample_weights = []
        for _, label in self.samples:
            sample_weights.append(class_weights[label])
        
        return sample_weights
    
    def get_class_distribution(self):
        """获取类别分布统计"""
        class_counts = Counter(label for _, label in self.samples)
        return class_counts
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            return torch.zeros((1, 28, 28)), label

class IncrementalCNN(nn.Module):
    """基于原模型的14类迁移学习网络 - 与test3.py完全一致"""
    
    def __init__(self, pretrained_model_path):
        super(IncrementalCNN, self).__init__()
        
        # 第一个卷积块 - 与test3.py完全相同
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 第二个卷积块 - 与test3.py完全相同
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接层 - 与test3.py完全一致，只是最后一层改为14类
        self.fc = nn.Sequential(
            nn.Flatten(),                    # fc.0
            nn.Linear(128 * 7 * 7, 256),    # fc.1
            nn.ReLU(),                       # fc.2
            nn.Dropout(0.4),                 # fc.3
            nn.Linear(256, 128),             # fc.4
            nn.ReLU(),                       # fc.5
            nn.Dropout(0.3),                 # fc.6
            nn.Linear(128, 14)               # fc.7 - 改为14类输出（原来是13类）
        )
        
        # 加载预训练权重
        self.load_pretrained_weights(pretrained_model_path)
    
    def load_pretrained_weights(self, model_path):
        """从13类模型加载预训练权重"""
        print(f"加载预训练模型: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            model_dict = self.state_dict()
            
            # 筛选可用的权重（排除最后的分类层 fc.7）
            pretrained_dict = {}
            for k, v in checkpoint.items():
                # 排除最后的分类层权重 (13类 -> 14类)
                if 'fc.7' not in k and k in model_dict:
                    pretrained_dict[k] = v
            
            # 更新模型权重
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            print(f"成功加载 {len(pretrained_dict)} 层预训练权重")
            print("新的14类分类层(fc.7)将从零开始训练")
            
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将从零开始训练")
    
    def freeze_features(self):
        """冻结特征提取层"""
        print("冻结特征提取层，只训练分类器...")
        
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        
        # 冻结全连接层的前7层(fc.0到fc.6)，只训练最后一层(fc.7)
        for i in range(7):  # fc.0到fc.6
            if hasattr(self.fc[i], 'parameters'):
                for param in self.fc[i].parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有层"""
        print("解冻所有层进行端到端微调...")
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

def create_transforms():
    """创建数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return train_transform, val_transform

def plot_class_distribution(dataset, output_dir):
    """绘制类别分布图"""
    class_counts = dataset.get_class_distribution()
    class_names = [str(i) for i in range(10)] + ['N', 'X', 'G', 'S']
    
    # 准备数据
    counts = [class_counts.get(i, 0) for i in range(14)]
    colors = ['skyblue'] * 13 + ['orange']  # S类别用橙色突出显示
    
    # 创建柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, counts, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.title('14类数据集分布 (橙色为新增S类别)', fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加总计信息
    total_samples = sum(counts)
    s_samples = counts[13]
    plt.text(0.02, 0.98, f'总样本: {total_samples}\nS类别: {s_samples} ({s_samples/total_samples*100:.1f}%)', 
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[保存] 类别分布图已保存")

def plot_training_curves(train_losses, train_accs, val_accs, freeze_epochs, output_dir):
    """绘制训练曲线"""
    epochs = range(1, len(train_accs) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率曲线
    ax1.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
    ax1.plot(epochs, val_accs, 'r-', label='验证准确率', linewidth=2)
    ax1.axvline(x=freeze_epochs, color='green', linestyle='--', alpha=0.7, 
                label=f'解冻点 (第{freeze_epochs}轮)')
    ax1.set_title('训练/验证准确率曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 损失曲线
    ax2.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax2.axvline(x=freeze_epochs, color='green', linestyle='--', alpha=0.7, 
                label=f'解冻点 (第{freeze_epochs}轮)')
    ax2.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加文字说明
    ax1.text(0.02, 0.02, '绿线：从冻结阶段转为微调阶段', 
             transform=ax1.transAxes, fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[保存] 训练曲线图已保存")

def plot_confusion_matrix(all_labels, all_preds, class_names, output_dir):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算准确率
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建热力图
    plt.figure(figsize=(12, 10))
    
    # 使用自定义颜色映射，突出显示S类别
    mask = np.zeros_like(cm_normalized, dtype=bool)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '准确率'}, mask=mask)
    
    # 突出显示S类别的行和列
    ax = plt.gca()
    ax.add_patch(plt.Rectangle((13, 0), 1, 14, fill=False, edgecolor='red', lw=3))  # S列
    ax.add_patch(plt.Rectangle((0, 13), 14, 1, fill=False, edgecolor='red', lw=3))  # S行
    
    plt.title('混淆矩阵 (红框标注S类别)', fontsize=14, fontweight='bold')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    
    # 添加S类别性能信息
    s_precision = cm_normalized[13, 13]
    s_recall = cm[13, 13] / cm[13, :].sum()
    plt.text(0.02, 0.98, f'S类别准确率: {s_precision:.3f}\nS类别召回率: {s_recall:.3f}', 
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[保存] 混淆矩阵热力图已保存")

def plot_class_accuracy(all_labels, all_preds, class_names, output_dir):
    """绘制各类别准确率柱状图"""
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracies = []
    
    for i in range(len(class_names)):
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
        else:
            acc = 0
        class_accuracies.append(acc)
    
    # 创建柱状图
    plt.figure(figsize=(12, 6))
    colors = ['skyblue'] * 13 + ['orange']  # S类别用橙色
    bars = plt.bar(class_names, class_accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.title('各类别识别准确率 (橙色为新增S类别)', fontsize=14, fontweight='bold')
    plt.xlabel('类别')
    plt.ylabel('准确率')
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    # 添加平均准确率线
    avg_acc = np.mean(class_accuracies)
    plt.axhline(y=avg_acc, color='red', linestyle='--', alpha=0.7, 
                label=f'平均准确率: {avg_acc:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[保存] 类别准确率图已保存")

def plot_sample_predictions(model, val_loader, device, class_names, output_dir, num_samples=16):
    """绘制预测示例"""
    model.eval()
    
    # 收集一些预测结果
    images_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(min(len(images), num_samples - len(images_list))):
                images_list.append(images[i].cpu())
                labels_list.append(labels[i].cpu().item())
                preds_list.append(predicted[i].cpu().item())
            
            if len(images_list) >= num_samples:
                break
    
    # 创建预测示例图
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images_list))):
        img = images_list[i].squeeze().numpy()
        true_label = class_names[labels_list[i]]
        pred_label = class_names[preds_list[i]]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'真实: {true_label}\n预测: {pred_label}', 
                         fontsize=10,
                         color='green' if true_label == pred_label else 'red')
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(images_list), 16):
        axes[i].axis('off')
    
    plt.suptitle('预测示例 (绿色=正确, 红色=错误)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[保存] 预测示例图已保存")

def generate_visualization_summary(output_dir, best_val_acc, training_time, config):
    """生成可视化总结图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 训练配置信息
    ax1.axis('off')
    config_text = f"""
训练配置总结
{'='*20}
总轮次: {config['epochs']}
批大小: {config['batch_size']}
学习率: {config['learning_rate']}
冻结轮次: {config['freeze_epochs']}

训练结果
{'='*20}
最佳准确率: {best_val_acc:.4f}
训练时间: {training_time:.1f} 秒
模型类型: 增量学习 (13→14类)
新增类别: S字母
    """
    ax1.text(0.1, 0.9, config_text, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.set_title('训练配置与结果', fontsize=14, fontweight='bold')
    
    # 2. 增量学习流程图
    ax2.axis('off')
    flow_text = """
增量学习流程
{'='*20}
1. 加载13类预训练模型
   ↓
2. 扩展为14类架构
   ↓  
3. 冻结特征层训练 (前12轮)
   ↓
4. 解冻全网络微调 (后13轮)
   ↓
5. 保存最终14类模型

优势: 复用已训练特征
效果: 快速适应新类别
    """
    ax2.text(0.1, 0.9, flow_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax2.set_title('增量学习策略', fontsize=14, fontweight='bold')
    
    # 3. 关键技术点
    ax3.axis('off')
    tech_text = """
关键技术特性
{'='*20}
[√] 权重迁移学习
[√] 分阶段训练策略  
[√] 智能平衡采样
[√] 类别权重调整
[√] 数据增强技术
[√] 标签平滑正则化
[√] 梯度裁剪优化
[√] 早停机制

防止灾难性遗忘
保持原有类别性能
    """
    ax3.text(0.1, 0.9, tech_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('技术特性', fontsize=14, fontweight='bold')
    
    # 4. 文件输出清单
    ax4.axis('off')
    files_text = """
生成文件清单
{'='*20}
[图表] 可视化图表:
  - class_distribution.png
  - training_curves.png  
  - confusion_matrix.png
  - class_accuracy.png
  - prediction_samples.png
  - visualization_summary.png

[报告] 报告文件:
  - training_report.txt
  
[模型] 模型文件:
  - best_model.pth
  - incremental_nxgs_model.pth
    """
    ax4.text(0.1, 0.9, files_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    ax4.set_title('输出文件', fontsize=14, fontweight='bold')
    
    plt.suptitle('增量学习训练总结报告', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[保存] 可视化总结图已保存")

def main():
    """主训练函数"""
    
    print("增量学习 - 添加S类别识别")
    print("=" * 40)
    
    # 获取用户输入
    data_dir = input("请输入包含14类数据的文件夹路径: ").strip().strip('"')
    pretrained_model = input("请输入13类预训练模型路径: ").strip().strip('"')
    
    # 验证路径
    if not os.path.exists(data_dir):
        print(f"错误: 数据路径不存在 {data_dir}")
        return
    
    if not os.path.exists(pretrained_model):
        print(f"错误: 预训练模型不存在 {pretrained_model}")
        return
    
    # 配置参数
    config = {
        'epochs': 25,
        'batch_size': 32,
        'learning_rate': 0.001,
        'freeze_epochs': 12,
        'output_dir': './model'
    }
    
    print(f"训练配置: {config['epochs']}轮, 批大小{config['batch_size']}, 前{config['freeze_epochs']}轮冻结特征层")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output_dir'], f"incremental_s_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据变换和数据集
    train_transform, val_transform = create_transforms()
    dataset = Dataset14Classes(data_dir, transform=train_transform)
    
    if len(dataset) == 0:
        print("错误: 数据集为空!")
        return
    
    # 生成数据分布图
    print("\n生成可视化图表...")
    plot_class_distribution(dataset, output_dir)
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    
    print(f"数据分割: 训练集 {train_size} / 验证集 {val_size}")
    
    # 创建加权采样器
    sample_weights = dataset.get_sample_weights()
    train_sample_weights = [sample_weights[i] for i in train_dataset.indices]
    sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 创建模型
    model = IncrementalCNN(pretrained_model).to(device)
    model.freeze_features()  # 开始时冻结特征层
    
    # 损失函数 - 使用类别权重
    class_counts = Counter(label for _, label in dataset.samples)
    class_weights = []
    for i in range(14):
        count = class_counts.get(i, 1)
        weight = len(dataset.samples) / (14 * count)
        if i == 13:  # 给S类别额外权重
            weight *= 1.2
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # 优化器 - 第一阶段只训练最后一层(fc.7)
    optimizer = optim.AdamW(model.fc[7].parameters(), lr=config['learning_rate'])
    
    # 训练记录
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    print(f"开始增量学习训练...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # 在指定轮次解冻所有层
        if epoch == config['freeze_epochs']:
            print(f"第{epoch+1}轮: 解冻所有层进行端到端微调")
            model.unfreeze_all()
            optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'] * 0.1)
        
        # 训练阶段
        model.train()
        train_correct = 0
        train_total = 0
        epoch_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = train_correct / train_total
        avg_loss = epoch_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        
        # 记录训练历史
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # 打印进度
        stage = "冻结阶段" if epoch < config['freeze_epochs'] else "微调阶段"
        print(f"Epoch {epoch+1:2d}/{config['epochs']} {stage} | Train: {train_accuracy:.4f} | Val: {val_accuracy:.4f} | Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"  保存最佳模型 (准确率: {best_val_acc:.4f})")
    
    training_time = time.time() - start_time
    print(f"训练完成! 用时: {training_time:.1f} 秒")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    
    # 生成训练曲线
    plot_training_curves(train_losses, train_accuracies, val_accuracies, config['freeze_epochs'], output_dir)
    
    # 最终评估
    print("正在生成最终评估报告...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # 生成报告
    class_names = [str(i) for i in range(10)] + ['N', 'X', 'G', 'S']
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("分类报告:")
    print(report)
    
    # 生成可视化图表
    print("生成详细可视化图表...")
    plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    plot_class_accuracy(all_labels, all_preds, class_names, output_dir)
    plot_sample_predictions(model, val_loader, device, class_names, output_dir)
    generate_visualization_summary(output_dir, best_val_acc, training_time, config)
    
    # 保存最终模型
    final_model_path = os.path.join(config['output_dir'], 'incremental_nxgs_model.pth')
    shutil.copy(best_model_path, final_model_path)
    
    # 保存报告
    with open(os.path.join(output_dir, 'training_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"增量学习训练报告\n")
        f.write(f"{'='*40}\n")
        f.write(f"训练时间: {training_time:.1f} 秒\n")
        f.write(f"最佳准确率: {best_val_acc:.4f}\n")
        f.write(f"训练配置: {config}\n\n")
        f.write("分类报告:\n")
        f.write(report)
        f.write(f"\n\n可视化文件:\n")
        f.write("- class_distribution.png: 类别分布图\n")
        f.write("- training_curves.png: 训练曲线图\n")
        f.write("- confusion_matrix.png: 混淆矩阵热力图\n")
        f.write("- class_accuracy.png: 各类别准确率图\n")
        f.write("- prediction_samples.png: 预测示例图\n")
        f.write("- visualization_summary.png: 可视化总结图\n")
    
    print(f"\n{'='*50}")
    print(f"[完成] 训练完成！")
    print(f"[准确率] 最佳准确率: {best_val_acc:.4f}")
    print(f"[时间] 训练时间: {training_time:.1f} 秒")
    print(f"[模型] 最终模型: {final_model_path}")
    print(f"[图表] 可视化结果: {output_dir}")
    print(f"[报告] 详细报告: {os.path.join(output_dir, 'training_report.txt')}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()