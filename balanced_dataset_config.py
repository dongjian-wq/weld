#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平衡数据集训练配置
适用于每个类别240张图片，总共3120张图片的平衡数据集
"""

# 数据集信息
DATASET_CONFIG = {
    "total_samples": 3120,  # 13类 × 240张/类
    "num_classes": 13,
    "samples_per_class": 240,
    "is_balanced": True,
    "class_names": [str(i) for i in range(10)] + ['N', 'X', 'G']
}

# 训练参数配置
TRAINING_CONFIG = {
    # 基础参数
    "batch_size": 64,           # 充足数据量可用较大批处理
    "epochs": 40,               # 平衡数据集减少训练轮次
    "learning_rate": 0.001,     # 标准学习率
    "weight_decay": 5e-5,       # 减少正则化强度
    
    # 优化器配置
    "optimizer": "adamw",
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    
    # 学习率调度
    "scheduler": "cosine",
    "min_lr_ratio": 0.01,
    
    # 正则化
    "dropout_rate": 0.3,        # 降低dropout率
    "label_smoothing": 0.1,     # 适度标签平滑
    "early_stopping": 10,       # 减少早停patience
    
    # 数据处理
    "use_weighted_sampling": False,  # 平衡数据不需要加权采样
    "use_class_weights": False,      # 平衡数据不需要类别权重
}

# 数据增强配置（适中强度）
AUGMENTATION_CONFIG = {
    "rotation_degrees": 20,     # 减少旋转角度
    "translate": (0.15, 0.15),  # 适中平移
    "scale": (0.8, 1.2),        # 适中缩放
    "shear": 10,                # 减少剪切
    "brightness": 0.3,          # 适中亮度变化
    "contrast": 0.3,            # 适中对比度变化
    "blur_prob": 0.2,           # 降低模糊概率
    "noise_prob": 0.15,         # 降低噪声概率
    "erase_prob": 0.2,          # 降低擦除概率
    "erase_scale": (0.02, 0.1), # 减少擦除范围
}

# 模型配置
MODEL_CONFIG = {
    "architecture": "enhanced_cnn",
    "num_classes": 13,
    "dropout_rate": 0.3,
    "use_batch_norm": True,
    "use_skip_connections": False,  # 平衡数据集可能不需要过于复杂的架构
}

# 验证配置
VALIDATION_CONFIG = {
    "val_ratio": 0.2,           # 20%用于验证
    "stratify": True,           # 分层采样保持类别平衡
    "shuffle": True,
}

# 输出配置
OUTPUT_CONFIG = {
    "save_best_model": True,
    "save_last_model": True,
    "save_training_curves": True,
    "save_confusion_matrix": True,
    "save_classification_report": True,
    "save_per_class_accuracy": True,
    "model_name": "balanced_nxg_model.pth",
}

# 推荐的命令行参数
RECOMMENDED_ARGS = {
    "batch_size": TRAINING_CONFIG["batch_size"],
    "epochs": TRAINING_CONFIG["epochs"],
    "learning_rate": TRAINING_CONFIG["learning_rate"],
    "weight_decay": TRAINING_CONFIG["weight_decay"],
    "dropout_rate": TRAINING_CONFIG["dropout_rate"],
    "label_smoothing": TRAINING_CONFIG["label_smoothing"],
    "early_stopping": TRAINING_CONFIG["early_stopping"],
    "use_weighted_sampling": False,
    "save_as": OUTPUT_CONFIG["model_name"],
}

def print_config_summary():
    """打印配置摘要"""
    print("="*60)
    print("平衡数据集训练配置摘要")
    print("="*60)
    print(f"数据集大小: {DATASET_CONFIG['total_samples']} 张图片")
    print(f"类别数量: {DATASET_CONFIG['num_classes']} 个")
    print(f"每类样本: {DATASET_CONFIG['samples_per_class']} 张")
    print(f"数据平衡: {'是' if DATASET_CONFIG['is_balanced'] else '否'}")
    print()
    print("推荐训练参数:")
    print(f"  批处理大小: {TRAINING_CONFIG['batch_size']}")
    print(f"  训练轮次: {TRAINING_CONFIG['epochs']}")
    print(f"  学习率: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Dropout率: {TRAINING_CONFIG['dropout_rate']}")
    print(f"  使用类别权重: {'否' if not TRAINING_CONFIG['use_class_weights'] else '是'}")
    print(f"  使用加权采样: {'否' if not TRAINING_CONFIG['use_weighted_sampling'] else '是'}")
    print("="*60)

def get_training_command(data_dir):
    """生成推荐的训练命令"""
    args = RECOMMENDED_ARGS
    cmd = f"""python train_balanced_dataset.py \\
    --data_dir "{data_dir}" \\
    --batch_size {args['batch_size']} \\
    --epochs {args['epochs']} \\
    --learning_rate {args['learning_rate']} \\
    --weight_decay {args['weight_decay']} \\
    --dropout_rate {args['dropout_rate']} \\
    --label_smoothing {args['label_smoothing']} \\
    --early_stopping {args['early_stopping']} \\
    --save_as {args['save_as']}"""
    
    if not args['use_weighted_sampling']:
        cmd += " \\\n    --no_weighted_sampling"
    
    return cmd

if __name__ == "__main__":
    print_config_summary()
    print("\n推荐训练命令:")
    print(get_training_command("path/to/your/dataset")) 