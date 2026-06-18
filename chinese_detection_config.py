# 汉字检测配置文件
# 您可以复制这些配置到 test3.py 中的 CHINESE_DETECTION_CONFIG 来调整参数

CHINESE_DETECTION_CONFIG = {
    # 基础参数
    'enable_chinese_detection': True,    # 是否启用汉字检测 (True/False)
    
    # 轮廓分析参数
    'min_contour_area': 100,            # 最小轮廓面积 (像素²) - 过滤小噪点
    'min_complexity': 2.0,              # 最小复杂度阈值 - 汉字笔画复杂
    'min_complexity_high': 2.5,         # 高复杂度阈值 - 用于第三个判断指标
    'min_complex_contours': 3,          # 最少复杂轮廓数量 - 一张图至少几个汉字
    
    # 字符尺寸参数
    'min_char_size': 20,                # 最小字符尺寸 (像素) - 汉字通常较大
    'aspect_ratio_min': 0.5,            # 最小宽高比 - 汉字接近方形
    'aspect_ratio_max': 2.0,            # 最大宽高比
    
    # 密度分析参数
    'density_min': 0.1,                 # 最小像素密度 - 汉字区域黑色像素比例
    'density_max': 0.6,                 # 最大像素密度
    'block_size': 50,                   # 分块大小 (像素) - 用于密度分析
    'min_high_density_regions': 5,      # 最少高密度区域数量
    
    # 综合判断参数
    'min_indicators': 2,                # 最少满足指标数 (总共3个指标)
}

# 参数调整建议:
# 
# 如果误识别太多数字为汉字 (过于敏感):
# - 增加 min_complex_contours (如改为 4 或 5)
# - 增加 min_complexity (如改为 2.5)
# - 增加 min_indicators (改为 3，需要满足所有指标)
# 
# 如果漏检汉字 (不够敏感):
# - 减少 min_complex_contours (如改为 2)
# - 减少 min_complexity (如改为 1.8)
# - 减少 min_indicators (改为 1，只需满足一个指标)
# 
# 完全关闭汉字检测:
# - 设置 enable_chinese_detection = False 