# Mask Replace Image

ComfyUI 自定义节点，用于智能检测、裁剪和替换图片中的物体。

## 节点列表

### 遮罩操作节点 (Mask)

#### MaskBoundingBox
从遮罩中提取边界框坐标。

**输入：** mask  
**输出：** left, top, right, bottom

#### MergeMasks
合并多个遮罩为一个遮罩（取并集）。

**输入：** masks  
**输出：** merged_mask

#### MergeMasksV2
合并多个遮罩（支持 9 个独立输入 + 批量输入）。

**输入：** mask_1 ~ mask_9, batched_masks(可选)  
**输出：** merged_mask

#### SelectLargestMask
根据 boxes 面积筛选出最大的遮罩。

**输入：** masks, boxes  
**输出：** largest_mask, index

#### SelectLargestMaskByArea
根据 mask 像素面积直接筛选出最大的遮罩（不依赖 bbox）。

**输入：** masks  
**输出：** largest_mask, index

---

### 图像处理节点 (Image)

#### CropImageByMask
根据遮罩边界框裁剪图片。

**输入：** image, mask  
**输出：** cropped_image, left, top, right, bottom

#### CropImageWithWhiteBackground
裁剪图片并将背景替换为白色，支持透明背景。

**输入：** image, mask  
**参数：** background_alpha (0.0=白色, 1.0=透明)  
**输出：** image(RGBA), mask

#### ReplaceBackgroundWithWhite
只替换背景为白色，不裁剪图片。

**输入：** image, mask  
**参数：** background_alpha (0.0=白色, 1.0=原图)  
**输出：** image

#### ReplaceBackgroundWithWhiteExpand
替换背景为白色，并可扩展空白区域以缩小遮罩物品占比。

**输入：** image, mask(可选)  
**参数：** expand_up/down/left/right, background_alpha  
**输出：** image(RGBA)

#### VisualizeDetectionBox
绘制遮罩边界框用于调试。

**输入：** image, mask  
**参数：** box_color, box_width  
**输出：** image

#### FillMaskWithColor
将遮罩区域填充成指定颜色。

**输入：** image, mask  
**参数：** R, G, B (0-255)  
**输出：** image

#### CropImageWithPosition
根据遮罩裁剪图像，支持四方向调整，输出裁剪信息和原图。

**输入：** image, mask  
**参数：** expand_up/down/left/right (像素扩展)  
**输出：** cropped_image, original_image, crop_position(JSON)

#### PasteCroppedImage
将处理后的裁剪图像贴回原图。

**输入：** processed_image, original_image, crop_position  
**参数：** feather (边缘羽化)  
**输出：** image

---

### 智能替换节点 (Replace)

#### ImageReplaceWithMask
智能物体替换核心节点（基础版）。

**输入：** base_image, base_mask, replace_image, replace_mask(可选)  
**参数：**
- keep_aspect_ratio: 保持宽高比
- cover_mode: 完全覆盖/完全适应
- alignment: 对齐方式 (center/top/bottom/left/right)
- feather: 边缘羽化

**输出：** image

#### ImageReplaceWithMaskV2
增强版替换节点，含偏移/缩放/裁切控制。

**新增参数：**
- offset_left/right/up/down: 四方向偏移
- allow_crop: 是否允许裁切
- scale_factor: 缩放比例(%)

#### ImageReplaceWithMaskV3
V3版，支持自适应扩展和白边控制。

**新增参数：**
- auto_expand_height: 高度自适应扩展
- auto_expand_width: 宽度自适应扩展
- enable_shrink_after_fit: 贴合后缩小
- shrink_ratio: 缩小比例 (0.01-1.0)

---

## 快速开始

### 基础替换流程
```
[原图] → [检测] → [分割] → [base_mask]
                              ↓
[替换图] → [预处理] --------→ [ImageReplaceWithMask] → [结果]
```

### 透明背景替换流程
```
[原图] → [检测] → [分割] → [base_mask]
                              ↓
[替换图] → [CropImageWithWhiteBackground] ──RGBA──┐
         (background_alpha=1.0 完全透明)          ▼
                              [ImageReplaceWithMask] → [结果]
```

### 裁剪-处理-贴回流程
```
[原图] ──┬──> [CropImageWithPosition] ──> [处理流程] ──> [PasteCroppedImage] ──> [结果]
         │           │                                        ▲
         │           └─────── crop_position ──────────────────┘
         └────────────────── original_image ──────────────────┘
```

---

## 安装

1. 放入 `ComfyUI/custom_nodes/` 目录
2. 安装依赖：`pip install torch numpy Pillow opencv-python`
3. 重启 ComfyUI

---

## 文件结构

```
Mask_Replace_Image/
├── __init__.py                    # 根目录入口
├── mask_nodes/                    # 遮罩和替换节点
│   ├── __init__.py
│   ├── utils.py                   # 工具函数
│   ├── mask_ops.py                # 遮罩操作节点
│   └── replace_ops.py             # 智能替换节点
├── image_nodes/                   # 图像处理节点
│   └── image_ops.py               # 图像处理节点
└── README.md                      # 本文件
```

---

## 依赖

- torch
- numpy
- Pillow
- opencv-python

---

## 许可证

MIT
