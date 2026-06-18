#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平衡数据集训练脚本
专为每个类别240张图片，总共3120张图片的平衡数据集优化
支持识别0-9数字和N、X、G字母（共13个类别）
"""

import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import argparse
from datetime import datetime
import shutil
import json
from collections import Counter

# 导入配置
try:
    from balanced_dataset_config import (
        DATASET_CONFIG, TRAINING_CONFIG, AUGMENTATION_CONFIG, 
        MODEL_CONFIG, VALIDATION_CONFIG, OUTPUT_CONFIG, 
        print_config_summary
    )
except ImportError:
    print("警告: 无法导入配置文件，将使用默认配置")

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

class BalancedDatasetHandler:
    """平衡数据集处理器"""
    
    @staticmethod
    def validate_dataset_balance(dataset, tolerance=0.1):
        """验证数据集是否平衡"""
        class_counts = Counter(label for _, label in dataset.samples)
        counts = list(class_counts.values())
        
        if not counts:
            return False, "数据集为空"
        
        mean_count = np.mean(counts)
        max_deviation = max(abs(count - mean_count) for count in counts)
        relative_deviation = max_deviation / mean_count
        
        is_balanced = relative_deviation <= tolerance
        
        return is_balanced, {
            "mean_samples_per_class": mean_count,
            "max_deviation": max_deviation,
            "relative_deviation": relative_deviation,
            "tolerance": tolerance,
            "class_counts": dict(class_counts)
        }
    
    @staticmethod
    def get_optimal_params_for_balanced_data(total_samples):
        """为平衡数据集获取最优参数"""
        if total_samples >= 3000:
            return {
                "batch_size": 64,
                "epochs": 40,
                "learning_rate": 0.001,
                "dropout_rate": 0.3,
                "weight_decay": 5e-5,
                "early_stopping": 10
            }
        elif total_samples >= 2000:
            return {
                "batch_size": 48,
                "epochs": 45,
                "learning_rate": 0.0008,
                "dropout_rate": 0.35,
                "weight_decay": 8e-5,
                "early_stopping": 12
            }
        else:
            return {
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.0005,
                "dropout_rate": 0.4,
                "weight_decay": 1e-4,
                "early_stopping": 15
            }

class FolderDigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        支持数字0-9和字母N、X、G的平衡数据集加载器
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 定义类别顺序：0-9数字 + N、X、G字母
        self.class_names = [str(i) for i in range(10)] + ['N', 'X', 'G']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # 收集所有样本
        for cls_name in self.class_names:
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                print(f"警告: 未找到类别文件夹 {cls_name}")
                continue
                
            label = self.class_to_idx[cls_name]
            
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(cls_path, img_file)
                    self.samples.append((img_path, label))
        
        # 验证数据集平衡性
        self.is_balanced, self.balance_info = BalancedDatasetHandler.validate_dataset_balance(self)
        self._print_dataset_info()
    
    def _print_dataset_info(self):
        """打印数据集信息"""
        class_counts = Counter(label for _, label in self.samples)
        print(f"\n数据集信息:")
        print(f"总样本数: {len(self.samples)}")
        print(f"数据集平衡: {'是' if self.is_balanced else '否'}")
        
        if not self.is_balanced:
            print(f"最大偏差: {self.balance_info['max_deviation']:.1f}")
            print(f"相对偏差: {self.balance_info['relative_deviation']:.2%}")
        
        print("各类别分布:")
        for i, cls_name in enumerate(self.class_names):
            count = class_counts.get(i, 0)
            percentage = count / len(self.samples) * 100 if self.samples else 0
            print(f"  {cls_name}: {count:3d} 张 ({percentage:5.1f}%)")
    
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
            image = torch.zeros((1, 28, 28))
            return image, label

class EnhancedCNNBalanced(nn.Module):
    """针对平衡数据集优化的CNN模型"""
    def __init__(self, num_classes=13, dropout_rate=0.3):
        super(EnhancedCNNBalanced, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

def create_balanced_transforms():
    """为平衡数据集创建适中强度的数据变换"""
    
    # 训练变换 - 适中强度
    train_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        # 适中的几何变换
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=15, 
            translate=(0.15, 0.15), 
            scale=(0.8, 1.2), 
            shear=10
        ),
        # 适中的颜色变换
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # 适度的随机效果
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ], p=0.2),
        transforms.ToTensor(),
        # 轻微的随机擦除
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 验证变换
    val_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return train_transform, val_transform

def _convert_to_serializable(obj):
    """将不可JSON序列化的对象转换为可序列化的格式"""
    if isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_serializable(item) for item in obj)
    elif hasattr(obj, 'item'):  # numpy类型
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy数组
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def create_visualizations(train_losses, val_losses, val_accuracies, 
                         confusion_mat, class_names, output_dir):
    """创建训练可视化图表"""
    
    # 训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='验证准确率', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('验证准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 每类别准确率
    plt.subplot(1, 3, 3)
    class_accuracies = confusion_mat.diagonal() / confusion_mat.sum(axis=1)
    bars = plt.bar(range(len(class_names)), class_accuracies)
    plt.xlabel('类别')
    plt.ylabel('准确率')
    plt.title('每个类别的准确率')
    plt.xticks(range(len(class_names)), class_names)
    
    # 为每个柱子添加数值标签
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_balanced_dataset(args):
    """平衡数据集训练函数"""
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出文件夹
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    output_dir = os.path.join(args.output_dir, f"balanced_training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出文件夹: {output_dir}")
    
    # 创建数据变换
    train_transform, val_transform = create_balanced_transforms()
    
    # 加载数据集
    print(f"加载数据集: {args.data_dir}")
    dataset = FolderDigitDataset(args.data_dir, transform=train_transform)
    
    if len(dataset) == 0:
        print("错误: 数据集为空!")
        return
    
    # 根据数据集大小获取最优参数
    optimal_params = BalancedDatasetHandler.get_optimal_params_for_balanced_data(len(dataset))
    
    # 如果用户使用默认参数，则采用优化参数
    if args.auto_optimize:
        args.batch_size = optimal_params["batch_size"]
        args.epochs = optimal_params["epochs"]
        args.learning_rate = optimal_params["learning_rate"]
        args.dropout_rate = optimal_params["dropout_rate"]
        args.weight_decay = optimal_params["weight_decay"]
        args.early_stopping = optimal_params["early_stopping"]
        print(f"自动优化参数: 批处理大小={args.batch_size}, 训练轮次={args.epochs}")
    
    # 使用分层采样保持每个类别的平衡
    try:
        from stratified_split_utils import BalancedDataLoader
        
        # 为训练数据创建一个临时数据集（用于划分）
        temp_dataset = FolderDigitDataset(args.data_dir, transform=None)
        
        # 使用分层采样创建平衡的DataLoader
        train_loader, val_loader, split_info = BalancedDataLoader.create_balanced_loaders(
            temp_dataset,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.num_workers,
            pin_memory=True,
            seed=args.seed
        )
        
        # 手动设置变换
        # 为训练集设置增强变换
        for idx in train_loader.dataset.indices:
            temp_dataset.samples[idx] = (temp_dataset.samples[idx][0], temp_dataset.samples[idx][1])
        
        # 重新创建带变换的数据集
        train_indices = train_loader.dataset.indices
        val_indices = val_loader.dataset.indices
        
        # 创建具有适当变换的训练和验证数据集
        train_dataset = FolderDigitDataset(args.data_dir, transform=train_transform)
        val_dataset = FolderDigitDataset(args.data_dir, transform=val_transform)
        
        # 创建子集
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        # 重新创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 设置变量以供后续使用
        train_size = split_info['train_samples']
        val_size = split_info['val_samples']
        
        print(f"✅ 使用分层采样划分数据集")
        print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
        
    except ImportError:
        print("⚠️  分层采样工具不可用，使用随机划分")
        # 回退到原始的随机划分方法
        train_size = int(args.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
        
        # 为验证集设置不同的变换
        val_dataset.dataset.transform = val_transform
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # 创建模型
    model = EnhancedCNNBalanced(num_classes=13, dropout_rate=args.dropout_rate).to(device)
    
    # 平衡数据集不需要类别权重
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # 训练记录
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    early_stop_counter = 0
    
    print(f"\n开始训练 - 总共 {args.epochs} 轮")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 轻微的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"  → 保存最佳模型 (准确率: {best_val_acc:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # 早停检查
        if args.early_stopping > 0 and early_stop_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {training_time:.2f} 秒")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # 最终评估
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # 计算混淆矩阵和分类报告
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [str(i) for i in range(10)] + ['N', 'X', 'G']
    classification_rep = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("\n分类报告:")
    print(classification_rep)
    
    # 创建可视化
    create_visualizations(train_losses, val_losses, val_accuracies, cm, class_names, output_dir)
    
    # 保存训练信息
    training_info = {
        "model": "EnhancedCNNBalanced",
        "dataset_size": int(len(dataset)),
        "dataset_balanced": bool(dataset.is_balanced),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "best_val_accuracy": float(best_val_acc),
        "training_time": float(training_time),
        "final_epoch": int(epoch + 1),
        "args": vars(args),
        "balance_info": _convert_to_serializable(dataset.balance_info),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    
    # 保存分类报告
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"训练时间: {training_time:.2f} 秒\n")
        f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
        f.write(f"数据集是否平衡: {'是' if dataset.is_balanced else '否'}\n\n")
        f.write("分类报告:\n")
        f.write(classification_rep)
    
    print(f"所有结果已保存到: {output_dir}")
    
    # 复制最佳模型
    if args.save_as:
        final_model_path = os.path.join(args.output_dir, args.save_as)
        shutil.copy(best_model_path, final_model_path)
        print(f"最佳模型已复制到: {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='平衡数据集手写字符识别训练程序')
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据集路径，包含0-9和N、X、G的文件夹')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集占总数据的比例')
    
    # 模型参数
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout率')
    
    # 训练参数（平衡数据集优化）
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=40,
                        help='训练轮次')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='权重衰减率')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑参数')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='早停patience')
    
    # 优化选项
    parser.add_argument('--auto_optimize', action='store_true', default=True,
                        help='自动根据数据集大小优化参数')
    parser.add_argument('--no_weighted_sampling', action='store_true', default=False,
                        help='不使用加权采样（平衡数据集推荐）')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='不使用CUDA')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作线程数')
    parser.add_argument('--output_dir', type=str, default='./model',
                        help='输出目录')
    parser.add_argument('--save_as', type=str, default='balanced_nxg_model.pth',
                        help='最佳模型保存名称')
    
    args = parser.parse_args()
    
    # 检查数据集路径
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据集路径不存在 {args.data_dir}")
        sys.exit(1)
    
    # 打印配置
    print("="*60)
    print("平衡数据集训练配置:")
    print(f"数据集路径: {args.data_dir}")
    print(f"自动优化参数: {'是' if args.auto_optimize else '否'}")
    print(f"批处理大小: {args.batch_size}")
    print(f"训练轮次: {args.epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"Dropout率: {args.dropout_rate}")
    print("="*60)
    
    # 开始训练
    train_balanced_dataset(args)

if __name__ == '__main__':
    main() 