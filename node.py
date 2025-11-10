import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple


def get_mask_bounding_box(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    从遮罩中提取边界框 (left, top, right, bottom)
    
    Args:
        mask: 遮罩张量 shape (H, W) 或 (1, H, W)
    
    Returns:
        (left, top, right, bottom) 边界框坐标
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
    # 转换为numpy数组
    mask_np = mask.cpu().numpy()
    
    # 找到所有非零像素的位置
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # 如果遮罩为空，返回零边界框
        return (0, 0, 0, 0)
    
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1]) - 1
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1]) - 1
    
    return (int(left), int(top), int(right), int(bottom))


def crop_image_by_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    background_alpha: float = 1.0
) -> torch.Tensor:
    """
    根据遮罩的边界框裁剪图片，并可选地处理背景透明度
    
    Args:
        image: 图片张量 shape (1, H, W, C)
        mask: 遮罩张量 shape (1, H, W) 或 (H, W)
        background_alpha: 背景透明度 (0.0=白色不透明, 1.0=保持原图)
    
    Returns:
        裁剪后的图片张量
    """
    # 确保 mask 是 3D 的 (1, H, W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    left, top, right, bottom = get_mask_bounding_box(mask)
    
    if left == right or top == bottom:
        # 如果边界框无效，返回原图
        return image
    
    # 裁剪图片和遮罩
    cropped = image[:, top:bottom+1, left:right+1, :].clone()
    cropped_mask = mask[:, top:bottom+1, left:right+1]
    
    # 处理背景：根据 background_alpha 将背景设置为白色
    # 确保 cropped_mask 的形状正确
    # cropped_mask shape: (1, H', W')
    # 需要扩展为 (1, H', W', 3) 以匹配 RGB 通道
    
    # 扩展维度：(1, H', W') -> (1, H', W', 1) -> (1, H', W', 3)
    # 使用 repeat 而不是 expand 确保创建新的张量
    mask_3ch = cropped_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
    
    # 创建白色背景 (值为1.0表示白色)
    white_background = torch.ones_like(cropped)
    
    # 背景混合公式：
    # - 物体区域 (mask≈1): 保持原始图片
    # - 背景区域 (mask≈0): 根据 background_alpha 混合
    #   background_alpha=0.0 -> 完全白色
    #   background_alpha=1.0 -> 保持原图
    # 
    # 完整公式: result = foreground + background
    # foreground = cropped * mask (保留物体)
    # background = (white * (1-alpha) + cropped * alpha) * (1-mask) (处理背景)
    result = (
        cropped * mask_3ch +  # 前景：保持物体原样
        (white_background * (1.0 - background_alpha) + cropped * background_alpha) * (1.0 - mask_3ch)  # 背景：混合
    )
    
    return result


def resize_image_to_fit(
    source_image: torch.Tensor,
    target_width: int,
    target_height: int,
    keep_aspect_ratio: bool = True,
    cover_mode: bool = False
) -> torch.Tensor:
    """
    调整图片大小以适配目标尺寸
    
    Args:
        source_image: 源图片张量 shape (1, H, W, C)
        target_width: 目标宽度
        target_height: 目标高度
        keep_aspect_ratio: 是否保持宽高比
        cover_mode: 覆盖模式 (True=完全覆盖可能裁剪, False=完全适应可能留空)
    
    Returns:
        调整大小后的图片张量
    """
    _, src_h, src_w, _ = source_image.shape
    
    if keep_aspect_ratio:
        # 计算缩放比例
        scale_w = target_width / src_w
        scale_h = target_height / src_h
        
        # cover_mode: True使用max确保完全覆盖, False使用min确保完全适应
        if cover_mode:
            scale = max(scale_w, scale_h)  # 完全覆盖，可能超出
        else:
            scale = min(scale_w, scale_h)  # 完全适应，可能留空
        
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
    else:
        new_w = target_width
        new_h = target_height
    
    # 转换为PIL图片
    img_np = source_image[0].cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    
    # 调整大小
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    
    # 转换回张量
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)[None,]
    
    return img_tensor


def composite_images(
    base_image: torch.Tensor,
    overlay_image: torch.Tensor,
    overlay_mask: torch.Tensor,
    target_bbox: Tuple[int, int, int, int],
    feather: int = 0,
    cover_mode: bool = False,
    alignment: str = "center"
) -> torch.Tensor:
    """
    将overlay图片合成到base图片的指定位置
    
    Args:
        base_image: 基础图片张量 shape (1, H, W, C)
        overlay_image: 覆盖图片张量 shape (1, H', W', C)
        overlay_mask: 覆盖图片的遮罩 shape (1, H', W')
        target_bbox: 目标位置边界框 (left, top, right, bottom)
        feather: 边缘羽化像素数
        cover_mode: 覆盖模式 (True=完全覆盖, False=完全适应)
        alignment: 对齐方式 (center/top/bottom/left/right)
    
    Returns:
        合成后的图片张量
    """
    left, top, right, bottom = target_bbox
    target_width = right - left + 1
    target_height = bottom - top + 1
    
    # 调整overlay图片大小以适配目标区域
    resized_overlay = resize_image_to_fit(
        overlay_image,
        target_width,
        target_height,
        keep_aspect_ratio=True,
        cover_mode=cover_mode
    )
    
    # 调整遮罩大小
    _, overlay_h, overlay_w, _ = resized_overlay.shape
    
    # 转换遮罩为PIL并调整大小
    mask_np = overlay_mask[0].cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)
    mask_pil = mask_pil.resize((overlay_w, overlay_h), Image.LANCZOS)
    
    # 应用羽化效果
    if feather > 0:
        from PIL import ImageFilter
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather))
    
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    resized_mask = torch.from_numpy(mask_np)[None,]
    
    # 创建输出图片（复制base图片）
    result = base_image.clone()
    
    # 处理覆盖模式：如果图片超出目标区域，需要裁剪
    if cover_mode and (overlay_w > target_width or overlay_h > target_height):
        # 根据对齐方式计算裁剪位置
        if alignment == "top":
            # 顶部对齐：从顶部开始，裁剪底部
            crop_top = 0
            crop_left = max(0, (overlay_w - target_width) // 2)
        elif alignment == "bottom":
            # 底部对齐：从底部开始，裁剪顶部
            crop_top = max(0, overlay_h - target_height)
            crop_left = max(0, (overlay_w - target_width) // 2)
        elif alignment == "left":
            # 左对齐：从左侧开始，裁剪右侧
            crop_left = 0
            crop_top = max(0, (overlay_h - target_height) // 2)
        elif alignment == "right":
            # 右对齐：从右侧开始，裁剪左侧
            crop_left = max(0, overlay_w - target_width)
            crop_top = max(0, (overlay_h - target_height) // 2)
        else:  # center (默认)
            # 居中对齐：居中裁剪
            crop_left = max(0, (overlay_w - target_width) // 2)
            crop_top = max(0, (overlay_h - target_height) // 2)
        
        crop_right = crop_left + min(overlay_w - crop_left, target_width)
        crop_bottom = crop_top + min(overlay_h - crop_top, target_height)
        
        # 裁剪图片和遮罩
        resized_overlay = resized_overlay[:, crop_top:crop_bottom, crop_left:crop_right, :]
        resized_mask = resized_mask[:, crop_top:crop_bottom, crop_left:crop_right]
        
        # 更新尺寸
        _, overlay_h, overlay_w, _ = resized_overlay.shape
        
        # 重新计算粘贴位置（裁剪后应该正好填满）
        paste_left = left
        paste_top = top
    else:
        # 计算粘贴位置（非覆盖模式或图片未超出）
        if alignment == "top":
            paste_top = top
            paste_left = left + (target_width - overlay_w) // 2
        elif alignment == "bottom":
            paste_top = top + target_height - overlay_h
            paste_left = left + (target_width - overlay_w) // 2
        elif alignment == "left":
            paste_left = left
            paste_top = top + (target_height - overlay_h) // 2
        elif alignment == "right":
            paste_left = left + target_width - overlay_w
            paste_top = top + (target_height - overlay_h) // 2
        else:  # center
            paste_left = left + (target_width - overlay_w) // 2
            paste_top = top + (target_height - overlay_h) // 2
        
        # 确保不超出边界
        paste_left = max(0, min(paste_left, base_image.shape[2] - overlay_w))
        paste_top = max(0, min(paste_top, base_image.shape[1] - overlay_h))
    
    # 扩展遮罩维度以匹配RGB通道
    mask_3ch = resized_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
    
    # 计算实际可以粘贴的区域（避免越界）
    paste_h = min(overlay_h, base_image.shape[1] - paste_top)
    paste_w = min(overlay_w, base_image.shape[2] - paste_left)
    
    # 合成图片
    result[
        :,
        paste_top:paste_top+paste_h,
        paste_left:paste_left+paste_w,
        :
    ] = (
        base_image[
            :,
            paste_top:paste_top+paste_h,
            paste_left:paste_left+paste_w,
            :
        ] * (1 - mask_3ch[:, :paste_h, :paste_w, :]) +
        resized_overlay[:, :paste_h, :paste_w, :] * mask_3ch[:, :paste_h, :paste_w, :]
    )
    
    return result


class MaskBoundingBox:
    """从遮罩提取边界框坐标"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    
    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("left", "top", "right", "bottom")
    
    def main(self, mask):
        """
        提取遮罩的边界框
        
        Args:
            mask: 输入遮罩
            
        Returns:
            (left, top, right, bottom) 边界框坐标
        """
        left, top, right, bottom = get_mask_bounding_box(mask)
        return (left, top, right, bottom)


class CropImageByMask:
    """根据遮罩边界框裁剪图片"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    
    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "left", "top", "right", "bottom")
    
    def main(self, image, mask):
        """
        根据遮罩裁剪图片
        
        Args:
            image: 输入图片
            mask: 输入遮罩
            
        Returns:
            裁剪后的图片和边界框坐标
        """
        cropped = crop_image_by_mask(image, mask)
        left, top, right, bottom = get_mask_bounding_box(mask)
        return (cropped, left, top, right, bottom)


class ImageReplaceWithMask:
    """
    根据遮罩智能替换图片物体
    
    这个节点可以：
    1. 检测原图中的物体并生成边界框
    2. 使用处理好的替换图（推荐先用 CropImageWithWhiteBackground 预处理）
    3. 自适应缩放替换物体
    4. 智能合成到原图
    
    注意：replace_image 推荐使用 CropImageWithWhiteBackground 节点预处理，
         已经裁剪好并添加了白色背景，可以直接使用
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "base_mask": ("MASK",),
                "replace_image": ("IMAGE",),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
                "cover_mode": ("BOOLEAN", {
                    "default": True,
                    "label_on": "完全覆盖",
                    "label_off": "完全适应"
                }),
                "alignment": (["center", "top", "bottom", "left", "right"], {
                    "default": "bottom"
                }),
                "feather": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "replace_mask": ("MASK",),
            }
        }
    
    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    
    def main(
        self,
        base_image,
        base_mask,
        replace_image,
        keep_aspect_ratio,
        cover_mode,
        alignment,
        feather,
        replace_mask=None
    ):
        """
        替换图片中的物体
        
        Args:
            base_image: 基础图片
            base_mask: 基础图片的遮罩（标识要替换的区域）
            replace_image: 替换源图片（推荐使用 CropImageWithWhiteBackground 预处理）
            keep_aspect_ratio: 是否保持宽高比
            cover_mode: 覆盖模式 (True=完全覆盖可能裁剪, False=完全适应可能留空)
            alignment: 对齐方式 (center/top/bottom/left/right)
            feather: 边缘羽化程度
            replace_mask: 可选的替换图片遮罩，如果不提供则使用整个图片
            
        Returns:
            合成后的图片
        """
        # 获取基础图片遮罩的边界框
        target_bbox = get_mask_bounding_box(base_mask)
        
        # 如果没有提供 replace_mask，创建一个全1的遮罩（使用整个图片）
        if replace_mask is None:
            _, h, w, _ = replace_image.shape
            replace_mask = torch.ones(1, h, w)
        
        # 确保 mask 是 3D 的 (1, H, W)
        if replace_mask.dim() == 2:
            replace_mask = replace_mask.unsqueeze(0)
        
        # 直接使用 replace_image（已经被 CropImageWithWhiteBackground 处理过）
        cropped_replace = replace_image
        
        # 使用 replace_mask 作为合成遮罩
        cropped_mask = replace_mask
        
        # 合成图片
        result = composite_images(
            base_image,
            cropped_replace,
            cropped_mask,
            target_bbox,
            feather,
            cover_mode=cover_mode,
            alignment=alignment
        )
        
        return (result,)


class CropImageWithWhiteBackground:
    """根据遮罩裁剪图片并将背景替换为白色"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "background_alpha": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    
    def main(self, image, mask, background_alpha):
        """
        根据遮罩裁剪图片并处理背景
        
        Args:
            image: 输入图片
            mask: 输入遮罩
            background_alpha: 背景透明度 (0.0=白色, 1.0=原图)
            
        Returns:
            处理后的图片和裁剪后的遮罩
        """
        # 确保 mask 是 3D 的 (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # 获取边界框
        left, top, right, bottom = get_mask_bounding_box(mask)
        
        if left == right or top == bottom:
            # 如果边界框无效，返回原图
            return (image, mask)
        
        # 裁剪图片和遮罩
        cropped_image = image[:, top:bottom+1, left:right+1, :].clone()
        cropped_mask = mask[:, top:bottom+1, left:right+1]
        
        # 扩展 mask 维度以匹配 RGB 通道
        mask_3ch = cropped_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        
        # 创建白色背景
        white_background = torch.ones_like(cropped_image)
        
        # 处理背景：
        # - 物体区域 (mask=1): 保持原图
        # - 背景区域 (mask=0): 根据 background_alpha 混合白色
        #   background_alpha=0.0 -> 完全白色
        #   background_alpha=1.0 -> 保持原图
        result_image = (
            cropped_image * mask_3ch +  # 前景：保持物体
            (white_background * (1.0 - background_alpha) + cropped_image * background_alpha) * (1.0 - mask_3ch)  # 背景：混合
        )
        
        return (result_image, cropped_mask)


class ReplaceBackgroundWithWhite:
    """只替换背景为白色，不裁剪图片"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "background_alpha": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    def main(self, image, mask, background_alpha):
        """
        只替换背景为白色，保持原图尺寸
        
        Args:
            image: 输入图片
            mask: 输入遮罩
            background_alpha: 背景透明度 (0.0=白色, 1.0=原图)
            
        Returns:
            处理后的图片（保持原始尺寸）
        """
        # 确保 mask 是 3D 的 (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # 克隆原图
        result_image = image.clone()
        
        # 扩展 mask 维度以匹配 RGB 通道
        mask_3ch = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        
        # 创建白色背景
        white_background = torch.ones_like(result_image)
        
        # 处理背景：
        # - 物体区域 (mask=1): 保持原图
        # - 背景区域 (mask=0): 根据 background_alpha 混合白色
        #   background_alpha=0.0 -> 完全白色
        #   background_alpha=1.0 -> 保持原图
        result_image = (
            result_image * mask_3ch +  # 前景：保持物体
            (white_background * (1.0 - background_alpha) + result_image * background_alpha) * (1.0 - mask_3ch)  # 背景：混合
        )
        
        return (result_image,)


class VisualizeDetectionBox:
    """在图片上可视化边界框"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "box_color": (["red", "green", "blue", "yellow", "white"], {"default": "red"}),
                "box_width": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
            }
        }
    
    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    
    def main(self, image, mask, box_color, box_width):
        """
        在图片上绘制遮罩边界框
        
        Args:
            image: 输入图片
            mask: 输入遮罩
            box_color: 边界框颜色
            box_width: 边界框线宽
            
        Returns:
            带边界框的图片
        """
        left, top, right, bottom = get_mask_bounding_box(mask)
        
        # 转换为PIL图片
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        # 绘制边界框
        draw = ImageDraw.Draw(pil_img)
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255)
        }
        color = color_map.get(box_color, (255, 0, 0))
        
        for i in range(box_width):
            draw.rectangle(
                [left-i, top-i, right+i, bottom+i],
                outline=color
            )
        
        # 转换回张量
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]
        
        return (img_tensor,)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskBoundingBox": MaskBoundingBox,
    "CropImageByMask": CropImageByMask,
    "ImageReplaceWithMask": ImageReplaceWithMask,
    "CropImageWithWhiteBackground": CropImageWithWhiteBackground,
    "ReplaceBackgroundWithWhite": ReplaceBackgroundWithWhite,
    "VisualizeDetectionBox": VisualizeDetectionBox,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBoundingBox": "提取遮罩边界框",
    "CropImageByMask": "按遮罩裁剪图片",
    "ImageReplaceWithMask": "智能物体替换",
    "CropImageWithWhiteBackground": "裁剪图片并替换背景为白色",
    "ReplaceBackgroundWithWhite": "只替换背景为白色",
    "VisualizeDetectionBox": "可视化检测框",
}






