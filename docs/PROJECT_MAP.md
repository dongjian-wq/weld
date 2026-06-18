# 项目梳理

## 一句话概览

本项目用于从焊接日报/表格图片中提取手写字符结果，并将正常结果、疑似异常结果、含汉字或需纠偏结果按目录分流保存。

## 代码分组

### OCR 推理流程

| 文件 | 作用 | 当前建议 |
| --- | --- | --- |
| `test3.py` | 当前主流程：YOLO 表格检测、单元格切分、预处理、字符识别、汉字检测、异常分流、Excel 输出 | 主入口，优先维护 |
| `test.py` | 早期流程：基于 OpenCV 表格线检测，不含 YOLO 主路径 | 可作为历史参考 |
| `test_14classes.py` | 加载 14 类模型，对单张图片做快速预测 | 保留为调试工具 |

### 模型训练

| 文件 | 作用 | 输入数据要求 |
| --- | --- | --- |
| `train_mnist_custom_dataset.py` | 早期自定义 MNIST 风格训练 | 按类别文件夹存放图片 |
| `train_balanced_dataset.py` | 平衡数据集训练，类别为 `0-9 + N/X/G` 共 13 类 | 子目录名为 `0` 到 `9`、`N`、`X`、`G` |
| `add_s_class_training.py` | 在 13 类模型基础上增量加入 `S` 类，共 14 类 | 子目录名为 `0` 到 `9`、`N`、`X`、`G`、`S` |
| `balanced_dataset_config.py` | 平衡数据集训练参数配置 | 被 `train_balanced_dataset.py` 使用 |

### 配置与模型

| 路径 | 说明 |
| --- | --- |
| `best.pt` | YOLO 表格检测模型，约 92MB |
| `model/incremental_nxgs_model.pth` | 当前字符识别模型，约 7.5MB |
| `chinese_detection_config.py` | 汉字检测参数参考，可用于调节 `test3.py` 内同名配置 |

## 数据与输出目录

| 目录 | 当前观察 | 建议 |
| --- | --- | --- |
| `data/` | 中间数据和训练/切分图片，约 12.4 万张 PNG，约 2.4GB | 不建议纳入 Git |
| `data/cell/` | 原始切分单元格或阶段性切分结果 | 可定期归档 |
| `data/processed_cells/` | 预处理后的单元格图片 | 可作为临时产物 |
| `处理结果/` | 正常 OCR 输出，包含 Excel、检测图、原图 | 按日期归档 |
| `待人工处理/` | 需要人工复核的输出 | 保留汇总 txt 和对应图片 |
| `待纠偏处理/` | 需要纠偏处理的输出 | 保留以便人工修正 |
| `Bad/` | 历史异常/过滤图片结果 | 可归档到外部存储 |

## 当前运行链路

```text
输入图片/目录
  -> test3.py
  -> YOLO(best.pt) 定位表格
  -> OpenCV 切分单元格
  -> 图像预处理，输出到 data/processed_cells/<timestamp>
  -> CNN(model/incremental_nxgs_model.pth) 识别字符
  -> 汉字/异常检测
  -> Excel + 可视化结果
  -> 正常结果、待人工处理、待纠偏处理分流
```

## 已发现的维护风险

1. `test3.py` 很大，超过 4000 行，多个职责混在一起，后续改动容易互相影响。
2. 运行路径存在硬编码，项目移动到 `D:\ai_code\...` 后可能找不到模型。
3. 模型结构在多个脚本中重复定义，训练和推理一旦不一致会导致权重加载失败。
4. 数据、模型、输出文件与源码混放，项目备份和版本管理成本高。
5. 代码注释在部分终端显示为乱码，建议统一保存为 UTF-8 并检查编辑器编码。

## 建议的目标结构

短期不用立刻搬动大文件，可以先按下面方向逐步迁移：

```text
.
├── src/
│   ├── pipeline.py          # OCR 主流程编排
│   ├── detection.py         # YOLO 表格检测
│   ├── table_split.py       # 表格线检测与单元格切分
│   ├── preprocess.py        # 单元格预处理
│   ├── models.py            # CNN 模型定义
│   ├── predict.py           # 模型加载与预测
│   ├── chinese_detect.py    # 汉字/非标准内容检测
│   └── exporters.py         # Excel 和可视化输出
├── scripts/
│   ├── run_ocr.py
│   ├── train_13class.py
│   └── train_14class_incremental.py
├── configs/
│   ├── ocr.yaml
│   └── training.yaml
├── models/
├── data/
├── outputs/
└── docs/
```

## 推荐优先级

1. 先改 `test3.py` 的模型路径参数化，解决换机器/换目录不能运行的问题。
2. 抽出模型定义，保证训练、测试、推理共用同一个 CNN 类。
3. 把 `test3.py` 拆出最容易独立的 Excel 输出和模型预测模块。
4. 新建一个 `samples/` 小样例，用于快速验证完整流程。
5. 最后再清理历史输出和大数据目录。
