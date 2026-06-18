import os
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

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']  # 优先使用的字体系列
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用sans-serif字体

# 自定义数据集类，用于加载按文件夹组织的图像数据集
class FolderDigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        支持数字和CSZDX五个字母的多类别数据集加载器
        :param root_dir: 数据集根目录，包含0-9和C、D、S、Z、X的子文件夹
        :param transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # 存储(图像路径, 标签)元组

        # 定义类别顺序和标签映射
        self.class_names = [str(i) for i in range(10)] + ['N', 'X', 'G']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        # 遍历所有类别文件夹
        for cls_name in self.class_names:
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            label = self.class_to_idx[cls_name]
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(cls_path, img_file)
                    self.samples.append((img_path, label))

        print(f"加载了 {len(self.samples)} 个样本，各类别分布:")
        # 统计每个类别的样本数量
        class_counts = {cls_name: 0 for cls_name in self.class_names}
        for _, label in self.samples:
            class_counts[self.class_names[label]] += 1
        for cls_name in self.class_names:
            print(f"类别 {cls_name}: {class_counts[cls_name]} 张图像")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')  # 转换为灰度图
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            image = torch.zeros((1, 28, 28))
            return image, label

# CNN模型定义 - 基本模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # -> [B, 32, 28, 28]
        x = self.pool(torch.relu(self.conv2(x)))  # -> [B, 64, 14, 14]
        x = self.pool(x)  # -> [B, 64, 7, 7]
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 增强版CNN模型 - 带BatchNorm和Dropout
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # 第一个卷积块 - 增加通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 第二个卷积块 - 增加深度和通道数
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接层 - 增加一层隐藏层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 13)  # 修改为13个类别（0-9数字 + NXG三个字母）
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# 添加一个更高级的模型 - ResNet风格
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 残差块1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # 残差块2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        # 下采样以匹配残差块2的输出
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )
        
        # 全局平均池化和全连接层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 初始卷积
        x = self.init_conv(x)
        
        # 残差块1 + 残差连接
        identity = x
        out = self.res_block1(x)
        out += identity
        out = self.relu(out)
        
        # 残差块2 + 残差连接（需要下采样）
        identity = self.downsample(out)
        out = self.res_block2(out)
        out += identity
        out = self.relu(out)
        
        # 全局池化和分类
        out = self.global_pool(out)
        out = self.fc(out)
        
        return out
        
# 自适应学习率调整的优化器
def create_optimizer(model, args):
    """创建优化器和学习率调度器"""
    # 提取不同层的参数组
    conv_params = []
    bn_params = []
    fc_params = []
    
    for name, param in model.named_parameters():
        if 'conv' in name:
            conv_params.append(param)
        elif 'bn' in name or 'batch_norm' in name:
            bn_params.append(param)
        elif 'fc' in name or 'linear' in name:
            fc_params.append(param)
    
    # 为不同层设置不同的学习率
    param_groups = [
        {'params': conv_params, 'weight_decay': args.weight_decay},
        {'params': bn_params, 'weight_decay': 0},  # BN层通常不使用权重衰减
        {'params': fc_params, 'weight_decay': args.weight_decay * 5}  # FC层使用更大的权重衰减
    ]
    
    # 创建优化器
    if args.optimizer == 'adam':
        return optim.Adam(param_groups, lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        return optim.AdamW(param_groups, lr=args.learning_rate)
    else:
        return optim.SGD(param_groups, lr=args.learning_rate, momentum=0.9, nesterov=True)

def create_output_folders(base_dir="./model"):
    """创建带时间戳的输出文件夹"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出文件夹: {output_dir}")
    return output_dir

def plot_training_curves(train_losses, val_losses, val_accuracies, output_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('训练轮次(Epoch)')
    plt.ylabel('损失值')
    plt.title('训练和验证损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='验证准确率')
    plt.xlabel('训练轮次(Epoch)')
    plt.ylabel('准确率')
    plt.title('验证准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, output_dir):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    class_names = [str(i) for i in range(10)] + ['N', 'X', 'G']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_sample_predictions(images, labels, predictions, output_dir):
    """绘制样本预测结果"""
    n_samples = min(25, len(images))
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))
    
    plt.figure(figsize=(12, 12))
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if predictions[i] == labels[i] else 'red'
        plt.title(f'预测: {predictions[i]}\n实际: {labels[i]}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close()

def plot_per_digit_accuracy(cm, output_dir):
    """绘制每个类别的识别准确率"""
    accuracies = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(accuracies)), accuracies)
    
    # 为每个柱子添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.3f}', ha='center', va='bottom')
    
    plt.xlabel('类别')
    plt.ylabel('准确率')
    plt.title('每个类别的识别准确率')
    plt.xticks(range(len(accuracies)), [str(i) for i in range(10)] + ['N', 'X', 'G'])
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_digit_accuracy.png'))
    plt.close()

def plot_prediction_distribution(all_predictions, output_dir):
    """绘制预测结果分布"""
    plt.figure(figsize=(10, 6))
    
    # 统计每个类别被预测的次数
    pred_counts = np.bincount(all_predictions, minlength=13)
    
    bars = plt.bar(range(13), pred_counts)
    
    # 为每个柱子添加数值标签
    for bar, count in zip(bars, pred_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count}', ha='center', va='bottom')
    
    plt.xlabel('预测的类别')
    plt.ylabel('预测次数')
    plt.title('模型预测分布')
    plt.xticks(range(13), [str(i) for i in range(10)] + ['N', 'X', 'G'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    plt.close()

def save_model_info(model, args, train_size, val_size, output_dir):
    """保存模型和训练信息"""
    info = {
        "model_type": model.__class__.__name__,
        "training_args": vars(args),
        "dataset_info": {
            "train_size": train_size,
            "val_size": val_size,
            "data_dir": args.data_dir
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存模型信息
    import json
    with open(os.path.join(output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

def calculate_class_weights(dataset):
    """计算类别权重以处理数据不平衡"""
    # 统计每个类别的样本数量
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.class_names[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # 计算类别权重（inverse frequency）
    total_samples = len(dataset.samples)
    num_classes = len(dataset.class_names)
    class_weights = []
    
    print("\n类别样本分布和权重:")
    for i, class_name in enumerate(dataset.class_names):
        count = class_counts.get(class_name, 0)
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 1.0  # 如果某个类别没有样本，设置默认权重
        class_weights.append(weight)
        print(f"类别 {class_name}: {count} 张图像, 权重: {weight:.3f}")
    
    return torch.FloatTensor(class_weights)

def train_and_evaluate(args):
    """训练和评估模型"""
    # 设置随机种子以确保结果可复现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出文件夹
    output_dir = create_output_folders(args.output_dir)
    
    # 定义数据增强变换（针对小数据集加强）
    train_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        # 增强几何变换
        transforms.RandomRotation(25),  # 增大旋转角度
        transforms.RandomAffine(
            degrees=15, 
            translate=(0.2, 0.2),  # 增大平移范围
            scale=(0.7, 1.3),      # 增大缩放范围
            shear=20               # 增大剪切角度
        ),
        # 增强颜色变换
        transforms.ColorJitter(brightness=0.4, contrast=0.4, gamma=0.2),
        # 随机应用更多增强
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ], p=0.3),  # 增加应用概率
        transforms.ToTensor(),
        # 随机添加噪声（在转为tensor后）
        transforms.RandomApply([
            lambda x: x + torch.randn_like(x) * 0.02
        ], p=0.2),
        # 更激进的随机擦除
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 创建混合增强策略
    def get_mixed_augmentation(apply_prob=0.5):
        """获取混合增强变换"""
        strong_aug = transforms.Compose([
            transforms.Resize((28, 28)),
            # 更激进的几何变换
            transforms.RandomRotation(35),
            transforms.RandomAffine(degrees=20, translate=(0.25, 0.25), scale=(0.6, 1.4), shear=25),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, gamma=0.3),
            # 更强的模糊和噪声
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
            ], p=0.4),
            transforms.ToTensor(),
            # 随机添加更多噪声
            transforms.RandomApply([
                lambda x: x + torch.randn_like(x) * 0.03
            ], p=0.3),
            # 更大范围的随机擦除
            transforms.RandomErasing(p=0.4, scale=(0.03, 0.2), ratio=(0.3, 3.3), value=0),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        weak_aug = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomRotation(15),  # 增强弱增强
            transforms.RandomAffine(degrees=8, translate=(0.12, 0.12), scale=(0.85, 1.15), shear=8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 根据概率决定使用强增强还是弱增强
        return lambda x: strong_aug(x) if np.random.random() < apply_prob else weak_aug(x)
    
    # 使用混合增强策略
    mixed_train_transform = get_mixed_augmentation(apply_prob=0.7)
    
    # 加载数据集
    print(f"正在加载数据集: {args.data_dir}")
    dataset = FolderDigitDataset(args.data_dir, transform=mixed_train_transform if args.use_mixed_aug else train_transform)
    
    # 计算类别权重以处理数据不平衡
    class_weights = calculate_class_weights(dataset).to(device)
    
    # 分割数据集
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 为验证集更改变换
    val_dataset.dataset.transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
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
    if args.model_type == 'simple':
        model = SimpleCNN().to(device)
    elif args.model_type == 'enhanced':
        model = EnhancedCNN().to(device)
    else:
        model = AdvancedCNN().to(device)
    
    # 在创建模型后加载预训练权重
    if args.pretrained:
        print(f"加载预训练模型: {args.pretrained}")
        pretrained_dict = torch.load(args.pretrained, map_location=device)
        
        # 处理模型结构不完全匹配的情况
        model_dict = model.state_dict()
        # 过滤不匹配的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
        
        # 更新当前模型的权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"成功加载了 {len(pretrained_dict)}/{len(model_dict)} 层预训练权重")
    
    # 定义损失函数 - 使用类别权重和标签平滑正则化
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"使用加权交叉熵损失函数处理数据不平衡问题")
    
    optimizer = create_optimizer(model, args)
    
    # 学习率调度器
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.5)
    elif args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # 训练循环
    print("开始训练...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 记录最佳性能模型对应的样本预测
    best_images = []
    best_labels = []
    best_preds = []
    
    # 实现早停策略
    early_stop_counter = 0
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用混合精度训练
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # 缩放梯度并优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准训练
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_preds = []
        val_images = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 记录标签和预测结果用于混淆矩阵
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                # 记录一些图像用于可视化
                if len(val_images) < 100:  # 最多记录100张图像
                    val_images.extend(images.cpu().numpy()[:min(10, len(images))])
                    
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        # 更新学习率
        if scheduler is not None:
            if args.scheduler == 'step' or args.scheduler == 'cosine':
                scheduler.step()
            elif args.scheduler == 'reduce':
                scheduler.step(val_loss)
        
        # 打印统计信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型到 {best_model_path}")
            
            # 保存当前的预测结果
            best_images = val_images[:25]  # 只保存前25张用于可视化
            best_labels = all_labels[:25]
            best_preds = all_preds[:25]
            
            # 重置早停计数器
            early_stop_counter = 0
        else:
            # 更新早停计数器
            early_stop_counter += 1
            
        # 检查是否应该早停
        if args.early_stopping > 0 and early_stop_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"训练完成！耗时: {training_time:.2f} 秒")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # 在验证集上评估最终性能
    final_correct = 0
    final_total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    final_accuracy = final_correct / final_total
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [str(i) for i in range(10)] + ['N', 'X', 'G']
    
    # 打印分类报告
    classification_rep = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n分类报告:")
    print(classification_rep)
    
    # 将分类报告保存到文件
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"训练时间: {training_time:.2f} 秒\n")
        f.write(f"最佳验证损失: {best_val_loss:.4f}\n")
        f.write(f"最终准确率: {final_accuracy:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(classification_rep)
    
    # 绘制训练曲线和混淆矩阵
    plot_training_curves(train_losses, val_losses, val_accuracies, output_dir)
    plot_confusion_matrix(cm, class_names, output_dir)
    plot_per_digit_accuracy(cm, output_dir)
    plot_prediction_distribution(all_preds, output_dir)
    
    # 可视化样本预测
    if best_images and best_labels and best_preds:
        plot_sample_predictions(best_images, best_labels, best_preds, output_dir)
    
    # 保存模型信息
    save_model_info(model, args, train_size, val_size, output_dir)
    
    # 复制最佳模型到主目录，使用有意义的名称
    if args.save_as:
        model_save_path = os.path.join(args.output_dir, args.save_as)
        shutil.copy(best_model_path, model_save_path)
        print(f"最佳模型已复制到: {model_save_path}")
    
    print(f"所有结果已保存到: {output_dir}")
    print(f"最终验证准确率: {final_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description='手写数字识别模型训练程序')
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, default=r'D:\OCR-project\weld_report\MNIST训练数据集\my_dataset',
                        help='数据集路径，包含0-9和C、D、S、Z、X的文件夹')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集占总数据的比例')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, choices=['simple', 'enhanced', 'advanced'], default='enhanced',
                        help='模型类型: simple=简单CNN, enhanced=增强CNN, advanced=高级CNN')
    
    # 训练参数（针对小数据集优化）
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小（小数据集适用）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮次（小数据集需要更多轮次）')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='学习率（小数据集适用较小学习率）')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adamw',
                        help='优化器类型（AdamW对小数据集效果更好）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减率（小数据集需要更强正则化）')
    parser.add_argument('--scheduler', type=str, choices=['step', 'reduce', 'cosine'], default='cosine',
                        help='学习率调度器类型（余弦退火适合小数据集）')
    parser.add_argument('--scheduler_step_size', type=int, default=10,
                        help='学习率调度器步长')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='不使用CUDA，即使可用')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器使用的工作线程数')
    parser.add_argument('--output_dir', type=str, default='./model',
                        help='输出目录')
    parser.add_argument('--save_as', type=str, default='custom_mnist_model.pth',
                        help='最佳模型的保存名称')
    parser.add_argument('--use_mixed_aug', action='store_true', default=True,
                        help='是否使用混合增强策略（小数据集推荐开启）')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='标签平滑正则化参数（小数据集适用较小值）')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='早停策略参数（小数据集需要更大patience）')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='是否使用混合精度训练')
    parser.add_argument('--pretrained', type=str, default='',
                        help='预训练模型路径，为空则从头训练')
    
    args = parser.parse_args()
    
    # 进行训练和评估
    train_and_evaluate(args)

if __name__ == '__main__':
    main() 