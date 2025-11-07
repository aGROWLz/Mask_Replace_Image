# Mask Replace Image

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个强大的 ComfyUI 自定义节点集合，用于智能检测、裁剪和替换图片中的物体。支持自动边界框提取、自适应缩放、智能合成等功能。

## ✨ 功能特性

- 🎯 **智能物体替换** - 基于遮罩的精确物体替换
- ✂️ **自动裁剪** - 根据遮罩边界框自动裁剪图片
- 🎨 **背景处理** - 支持背景替换为白色（裁剪或非裁剪模式）
- 📐 **自适应缩放** - 支持保持宽高比或拉伸填充
- 🎭 **多种对齐方式** - 支持顶部、底部、居中、左右对齐
- 🔍 **可视化调试** - 边界框可视化工具
- 🪶 **边缘羽化** - 平滑的边缘过渡效果

## 📦 安装

### 方法1: Git Clone（推荐）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/Mask_Replace_Image.git
```

### 方法2: 手动安装

1. 下载或克隆此仓库
2. 将整个文件夹放入 `ComfyUI/custom_nodes/` 目录
3. 重启 ComfyUI

## 🚀 快速开始

### 基本工作流

```
原始图片 → GroundingDINO检测 → SAM分割 → base_mask
替换图片 → GroundingDINO检测 → SAM分割 → replace_mask
                                                      ↓
替换图片 + replace_mask → CropImageWithWhiteBackground → 处理后的图片
                                                      ↓
原始图片 + base_mask + 处理后的图片 → ImageReplaceWithMask → 输出图片
```

### 示例：替换桌面上的物体

1. **检测原图中的物体**
   - 使用 `GroundingDinoSAMSegment` 检测目标物体
   - 输入 prompt: `"product"` 或具体物体名
   - 获得 `base_image` 和 `base_mask`

2. **预处理替换图片**
   - 使用 `CropImageWithWhiteBackground` 处理替换源
   - 设置 `background_alpha: 0.0` 将背景变为白色

3. **执行替换**
   - 使用 `ImageReplaceWithMask` 进行智能替换
   - 推荐参数：
     - `cover_mode: True` (完全覆盖)
     - `alignment: bottom` (底部对齐，适合桌面物体)
     - `feather: 5-10` (边缘羽化)

## 📚 节点说明

### 1. MaskBoundingBox - 提取遮罩边界框

从遮罩中提取最小包围矩形边界框坐标。

**输入:**
- `mask` (MASK): 输入遮罩

**输出:**
- `left` (INT): 左边界坐标
- `top` (INT): 上边界坐标
- `right` (INT): 右边界坐标
- `bottom` (INT): 下边界坐标

**使用场景:**
- 获取物体位置信息
- 用于其他节点的坐标输入

---

### 2. CropImageByMask - 按遮罩裁剪图片

根据遮罩的边界框自动裁剪图片。

**输入:**
- `image` (IMAGE): 输入图片
- `mask` (MASK): 遮罩

**输出:**
- `cropped_image` (IMAGE): 裁剪后的图片
- `left, top, right, bottom` (INT): 边界框坐标

**使用场景:**
- 提取物体区域
- 获取裁剪后的图片和坐标信息

---

### 3. ImageReplaceWithMask - 智能物体替换 ⭐

核心功能节点，实现智能物体替换。

**输入:**
- `base_image` (IMAGE): 基础图片（要被替换的图片）
- `base_mask` (MASK): 基础图片的遮罩（标识要替换的区域）
- `replace_image` (IMAGE): 替换源图片（推荐先用 `CropImageWithWhiteBackground` 预处理）
- `replace_mask` (MASK, 可选): 替换图片的遮罩，如果不提供则使用整个图片
- `keep_aspect_ratio` (BOOLEAN, 默认: True): 是否保持宽高比
- `cover_mode` (BOOLEAN, 默认: True): 覆盖模式
  - `True`: 完全覆盖目标区域，可能裁剪超出部分 ✅ 推荐
  - `False`: 完全适应目标区域，可能留有空白
- `alignment` (STRING, 默认: "bottom"): 对齐方式
  - `"bottom"`: 底部对齐（推荐，适合桌面物体）
  - `"top"`: 顶部对齐（适合悬挂物体）
  - `"center"`: 居中对齐
  - `"left"`: 左对齐
  - `"right"`: 右对齐
- `feather` (INT, 默认: 5, 范围: 0-100): 边缘羽化程度

**输出:**
- `image` (IMAGE): 合成后的图片

**特性:**
- ✅ 自动提取目标区域边界框
- ✅ 自适应缩放（保持比例或拉伸）
- ✅ 多种对齐方式
- ✅ 边缘羽化，无缝融合
- ✅ 防止尺寸不匹配问题

**推荐工作流:**
1. 使用 `CropImageWithWhiteBackground` 预处理替换源
2. 将处理后的图片和遮罩直接输入此节点
3. 享受简洁高效的替换效果

---

### 4. CropImageWithWhiteBackground - 裁剪并替换背景为白色

根据遮罩裁剪图片，并将背景区域替换为白色，物体区域保持不变。

**输入:**
- `image` (IMAGE): 输入图片
- `mask` (MASK): 遮罩（标识物体区域）
- `background_alpha` (FLOAT, 默认: 0.0, 范围: 0.0-1.0): 背景透明度控制
  - `0.0`: 背景完全变为白色（推荐）
  - `0.5`: 背景半透明白化
  - `1.0`: 背景保持原图

**输出:**
- `image` (IMAGE): 处理后的图片（背景变白）
- `mask` (MASK): 裁剪后的遮罩

**特性:**
- ✅ 根据遮罩边界框裁剪图片
- ✅ 将背景区域替换为白色
- ✅ 物体区域保持原样（不透明）
- ✅ 可控背景透明度

**使用场景:**
- 预处理替换源图片，去除杂乱背景
- 提取产品图并添加白色背景
- 为 `ImageReplaceWithMask` 准备干净的替换源

---

### 5. ReplaceBackgroundWithWhite - 替换背景为白色（不裁剪）🆕

将图片背景替换为白色，保持原图尺寸不变。

**输入:**
- `image` (IMAGE): 输入图片
- `mask` (MASK): 遮罩（标识物体区域）
- `background_alpha` (FLOAT, 默认: 0.0, 范围: 0.0-1.0): 背景透明度控制
  - `0.0`: 背景完全变为白色（推荐）
  - `0.5`: 背景半透明白化
  - `1.0`: 背景保持原图

**输出:**
- `image` (IMAGE): 处理后的图片（背景变白）
- `mask` (MASK): 原始遮罩（不裁剪）

**特性:**
- ✅ 保持原图尺寸，不进行裁剪
- ✅ 将背景区域替换为白色
- ✅ 物体区域保持原样（不透明）
- ✅ 自动处理遮罩尺寸不匹配的情况

**使用场景:**
- 需要保持原图尺寸的场景
- 仅替换背景，不需要裁剪
- 与 `CropImageWithWhiteBackground` 的区别：不裁剪图片

---

### 6. VisualizeDetectionBox - 可视化检测框

在图片上绘制遮罩的边界框，用于调试和预览。

**输入:**
- `image` (IMAGE): 输入图片
- `mask` (MASK): 遮罩
- `box_color` (STRING, 默认: "red"): 边界框颜色
  - `"red"`, `"green"`, `"blue"`, `"yellow"`, `"white"`
- `box_width` (INT, 默认: 3, 范围: 1-20): 边界框线宽

**输出:**
- `image` (IMAGE): 带边界框的图片

**使用场景:**
- 调试检测结果
- 预览边界框位置
- 验证遮罩准确性

## 💡 使用示例

### 示例1: 电商产品替换

**目标**: 将桌面上的旧款手机替换为新款手机

**工作流:**
```
桌面场景图片 → GroundingDINO检测 → SAM分割 → base_mask
新手机产品图 → GroundingDINO检测 → SAM分割 → replace_mask
                                                      ↓
新手机产品图 + replace_mask → CropImageWithWhiteBackground → 处理后的图片
                                                      ↓
桌面场景图片 + base_mask + 处理后的图片 → ImageReplaceWithMask → 输出图片
```

**推荐参数:**
- `keep_aspect_ratio: True`
- `cover_mode: True`
- `alignment: bottom`
- `feather: 10-15`

### 示例2: 服装替换

**目标**: 替换模特身上的T恤

**工作流:**
```
原始图片 → 检测 "t-shirt" → base_mask
替换图片 → 检测 "t-shirt" → replace_mask
替换图片 + replace_mask → CropImageWithWhiteBackground → 处理后的图片
原始图片 + base_mask + 处理后的图片 → ImageReplaceWithMask → 输出图片
```

**推荐参数:**
- `keep_aspect_ratio: True`
- `cover_mode: True`
- `alignment: center`
- `feather: 20` (服装需要更柔和的边缘)

### 示例3: 仅替换背景（不裁剪）

**目标**: 将产品图的背景替换为白色，保持原图尺寸

**工作流:**
```
产品图片 + 遮罩 → ReplaceBackgroundWithWhite → 输出图片
```

**推荐参数:**
- `background_alpha: 0.0` (完全白色背景)

## 🔧 参数调优指南

### cover_mode (覆盖模式)

- **True（完全覆盖）** - 推荐 ✅
  - 缩放替换图以完全覆盖目标区域
  - 不会露出原图物体
  - 可能裁剪超出部分

- **False（完全适应）**
  - 缩放替换图完全适应目标区域
  - 替换图完整显示
  - 可能留有空白

### alignment (对齐方式)

- **bottom** - 默认，推荐 ✅
  - 适合桌面/地面物体
  - 保留底边，裁剪顶边

- **top**
  - 适合悬挂物体
  - 保留顶边，裁剪底边

- **center**
  - 适合居中物体
  - 均衡裁剪

### feather (边缘羽化)

- `0-3`: 几乎无羽化，适合硬边缘物体
- `5-10`: 轻微羽化，适合一般物体 ✅ 推荐
- `15-30`: 中度羽化，适合需要柔和过渡的场景
- `30+`: 强烈羽化，适合艺术效果

### background_alpha (背景透明度)

- `0.0`: 背景完全变为白色 ✅ 推荐
- `0.1-0.3`: 背景部分白化
- `0.5-0.9`: 背景轻微淡化
- `1.0`: 保持原始背景

## 📖 详细文档

更多详细的使用指南、参数说明和常见问题，请查看 [USAGE_GUIDE.md](USAGE_GUIDE.md)。

## 🛠️ 依赖项

- `torch` - PyTorch深度学习框架
- `numpy` - 数值计算库
- `PIL` (Pillow) - 图像处理库
- ComfyUI 核心

## ⚠️ 注意事项

1. **遮罩质量**: 遮罩质量直接影响替换效果，建议使用 SAM 等高质量分割模型
2. **尺寸差异**: 节点会自动处理尺寸差异，但极端比例可能导致变形
3. **羽化参数**: 羽化值过大可能导致边缘模糊，建议 5-15 之间
4. **性能**: 大图片处理可能较慢，建议先调整图片大小

## 📝 版本历史

### v1.6.0 (最新)
- ✨ 新增 `ReplaceBackgroundWithWhite` 节点
- 支持不裁剪图片，仅替换背景为白色
- 自动处理遮罩尺寸不匹配的情况

### v1.5.0
- ✨ 添加对齐方式参数
- 支持 `bottom`, `top`, `center`, `left`, `right` 对齐
- 解决完全覆盖模式下裁掉重要部分的问题

### v1.4.0
- ✨ 添加覆盖模式参数
- 新增 `cover_mode` 控制缩放行为
- 解决替换后露出原图物体边缘的问题

### v1.3.0
- 🔧 简化工作流
- `replace_mask` 改为可选参数
- 推荐使用 `CropImageWithWhiteBackground` 预处理

### v1.2.0
- ✨ 新增 `CropImageWithWhiteBackground` 节点
- 独立处理替换源图片背景
- 支持预览和调试处理后的结果

### v1.1.0
- ✨ 背景透明度控制
- 支持将背景设置为白色或保持原图

### v1.0.0
- 🎉 初始版本
- 基础边界框提取
- 智能物体替换
- 自适应缩放
- 边缘羽化

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**Made with ❤️ for ComfyUI Community**
