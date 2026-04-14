# Mask Replace Image

ComfyUI 自定义节点，用于智能检测、裁剪和替换图片中的物体。

## 节点列表

### MaskBoundingBox
从遮罩中提取边界框坐标。

**输入：** mask  
**输出：** left, top, right, bottom

### CropImageByMask
根据遮罩边界框裁剪图片。

**输入：** image, mask  
**输出：** cropped_image, left, top, right, bottom

### ImageReplaceWithMask
智能物体替换核心节点。

**输入：** base_image, base_mask, replace_image, replace_mask(可选)  
**参数：** keep_aspect_ratio, cover_mode, alignment, feather  
**输出：** image

### ImageReplaceWithMaskV3
增强版替换节点，支持自适应扩展和白边控制。

**新增参数：** offset_*, auto_expand_*, enable_shrink_after_fit, shrink_ratio

### CropImageWithWhiteBackground
裁剪图片并将背景替换为白色。

**输入：** image, mask  
**参数：** background_alpha (0.0-1.0)  
**输出：** image, mask

### VisualizeDetectionBox
绘制遮罩边界框用于调试。

**输入：** image, mask  
**参数：** box_color, box_width  
**输出：** image

### MergeMasksV2
合并多个遮罩（支持 9 个独立输入 + 批量输入）。

**输入：** mask_1 ~ mask_9, batched_masks(可选)  
**输出：** merged_mask

## 快速开始

```
[原图] → [检测] → [分割] → [base_mask]
                              ↓
[替换图] → [预处理] --------→ [ImageReplaceWithMask] → [结果]
```

## 安装

1. 放入 `ComfyUI/custom_nodes/`
2. 重启 ComfyUI

## 依赖

torch, numpy, Pillow, opencv-python

## 许可证

MIT
