#!/usr/bin/env python3
"""
14类手写字符识别测试脚本
支持识别：0-9数字 + N、X、G、S字母
"""

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import argparse

def load_14class_model(model_path):
    """加载14类模型"""
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 14)  # 14类
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.fc(x)
            return x

    model = CNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"成功加载14类模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    model.eval()
    return model

def predict_image(model, image_path):
    """预测图像"""
    class_names = [str(i) for i in range(10)] + ['N', 'X', 'G', 'S']
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        image = Image.open(image_path).convert('L')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return class_names[predicted.item()], confidence.item()
    
    except Exception as e:
        print(f"预测失败: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='14类手写字符识别测试')
    parser.add_argument('input_path', type=str, help='输入图像的路径或包含图像的目录')
    parser.add_argument('--model', type=str, default='./model/incremental_nxgs_model.pth')
    
    args = parser.parse_args()
    
    print("14类手写字符识别测试")
    print("支持: 0-9数字 + N、X、G、S字母")
    print("=" * 40)
    
    model = load_14class_model(args.model)
    if model is None:
        return
    
    predicted_class, confidence = predict_image(model, args.input_path)
    
    if predicted_class is not None:
        print(f"输入图像: {args.input_path}")
        print(f"预测结果: {predicted_class}")
        print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")

if __name__ == '__main__':
    main()