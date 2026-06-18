# Weld Report OCR Project

这是一个面向焊接日报/表格图片的 OCR 项目，核心流程是：

1. 使用 YOLO 定位日报图片中的表格区域。
2. 基于表格线切分单元格。
3. 对单元格做去边框、二值化、缩放等预处理。
4. 使用 PyTorch CNN 模型识别手写字符：`0-9`、`N`、`X`、`G`、`S`。
5. 将识别结果输出为 Excel、检测可视化图片，并把疑似异常/含汉字图片分流到人工处理目录。

## 当前主入口

推荐优先看和运行：

```powershell
python test3.py "待处理图片或图片目录"
```

`test3.py` 是当前功能最完整的处理脚本，包含 YOLO 表格检测、单元格切分、字符识别、汉字检测、异常图片分流和结果输出。

注意：脚本中还有硬编码的模型路径，例如：

```python
r'D:\OCR-project\weld_report\best.pt'
r'D:\OCR-project\weld_report\model\incremental_nxgs_model.pth'
```

如果当前项目实际路径是 `D:\ai_code\OCR-project\weld_report`，运行前需要把这些路径改成当前目录，或后续重构为命令行参数。

## 目录说明

```text
.
├── test3.py                         # 当前增强版 OCR 主流程
├── test.py                          # 早期 OCR 流程，未使用 YOLO
├── train_balanced_dataset.py         # 13 类模型训练：0-9 + N/X/G
├── add_s_class_training.py           # 增量训练，加入 S 类形成 14 类
├── test_14classes.py                 # 单张图片 14 类模型测试
├── train_mnist_custom_dataset.py      # 早期 MNIST 风格训练脚本
├── balanced_dataset_config.py         # 平衡数据集训练配置
├── chinese_detection_config.py        # 汉字检测参数参考配置
├── best.pt                           # YOLO 表格检测模型
├── model/
│   └── incremental_nxgs_model.pth     # 当前字符识别模型
├── data/                              # 原始/中间数据，体量较大
├── 处理结果/                           # OCR 正常输出结果
├── 待人工处理/                         # 需要人工复核的输出
├── 待纠偏处理/                         # 需要纠偏复核的输出
├── Bad/                               # 历史异常/过滤结果
└── docs/
    └── PROJECT_MAP.md                 # 更详细的项目梳理
```

## 环境依赖

先安装 `requirements.txt` 中的通用依赖：

```powershell
pip install -r requirements.txt
```

如果需要 GPU/CUDA 版本的 PyTorch，请按本机 CUDA 版本从 PyTorch 官网安装对应命令，再安装其余依赖。

## 常用命令

运行 OCR：

```powershell
python test3.py "D:\path\to\images"
```

训练 13 类字符模型：

```powershell
python train_balanced_dataset.py --data_dir "D:\path\to\dataset" --output_dir ".\model" --save_as balanced_nxg_model.pth
```

测试 14 类字符模型：

```powershell
python test_14classes.py "D:\path\to\cell.png" --model ".\model\incremental_nxgs_model.pth"
```

## 建议的后续整理方向

1. 把 `test3.py` 中的硬编码模型路径改为命令行参数。
2. 把模型结构抽到 `src/models.py`，避免训练和推理脚本重复定义 CNN。
3. 把图像切表、预处理、Excel 输出、异常分流拆成独立模块。
4. 将 `data/`、`处理结果/`、`待人工处理/` 等大文件目录排除在 Git 外。
5. 增加一个小样例目录和最小可运行测试，避免每次验证都扫完整大数据集。
