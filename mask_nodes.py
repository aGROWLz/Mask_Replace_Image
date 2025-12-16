import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Tuple
import json
import ast


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
    将overlay图片合成到base图片的指定位置（基础版，与bak一致）
    """
    left, top, right, bottom = target_bbox
    target_width = right - left + 1
    target_height = bottom - top + 1
    
    resized_overlay = resize_image_to_fit(
        overlay_image,
        target_width,
        target_height,
        keep_aspect_ratio=True,
        cover_mode=cover_mode
    )
    
    _, overlay_h, overlay_w, _ = resized_overlay.shape
    
    mask_np = overlay_mask[0].cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)
    mask_pil = mask_pil.resize((overlay_w, overlay_h), Image.LANCZOS)
    
    if feather > 0:
        from PIL import ImageFilter
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather))
    
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    resized_mask = torch.from_numpy(mask_np)[None,]
    
    result = base_image.clone()
    
    if cover_mode and (overlay_w > target_width or overlay_h > target_height):
        if alignment == "top":
            crop_top = 0
            crop_left = max(0, (overlay_w - target_width) // 2)
        elif alignment == "bottom":
            crop_top = max(0, overlay_h - target_height)
            crop_left = max(0, (overlay_w - target_width) // 2)
        elif alignment == "left":
            crop_left = 0
            crop_top = max(0, (overlay_h - target_height) // 2)
        elif alignment == "right":
            crop_left = max(0, overlay_w - target_width)
            crop_top = max(0, (overlay_h - target_height) // 2)
        else:
            crop_left = max(0, (overlay_w - target_width) // 2)
            crop_top = max(0, (overlay_h - target_height) // 2)
        
        crop_right = crop_left + min(overlay_w - crop_left, target_width)
        crop_bottom = crop_top + min(overlay_h - crop_top, target_height)
        
        resized_overlay = resized_overlay[:, crop_top:crop_bottom, crop_left:crop_right, :]
        resized_mask = resized_mask[:, crop_top:crop_bottom, crop_left:crop_right]
        
        _, overlay_h, overlay_w, _ = resized_overlay.shape
        
        paste_left = left
        paste_top = top
    else:
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
        else:
            paste_left = left + (target_width - overlay_w) // 2
            paste_top = top + (target_height - overlay_h) // 2
        
        paste_left = max(0, min(paste_left, base_image.shape[2] - overlay_w))
        paste_top = max(0, min(paste_top, base_image.shape[1] - overlay_h))
    
    mask_3ch = resized_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
    
    paste_h = min(overlay_h, base_image.shape[1] - paste_top)
    paste_w = min(overlay_w, base_image.shape[2] - paste_left)
    
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


def composite_images_v2(
    base_image: torch.Tensor,
    overlay_image: torch.Tensor,
    overlay_mask: torch.Tensor,
    target_bbox: Tuple[int, int, int, int],
    feather: int = 0,
    cover_mode: bool = False,
    alignment: str = "center",
    offset_x: int = 0,
    offset_y: int = 0,
    allow_crop: bool = True,
    scale_factor: float = 0.0
) -> torch.Tensor:
    """
    将overlay图片合成到base图片的指定位置（增强版，含偏移/缩放/裁切）
    """
    left, top, right, bottom = target_bbox
    target_width = right - left + 1
    target_height = bottom - top + 1
    
    resized_overlay = resize_image_to_fit(
        overlay_image,
        target_width,
        target_height,
        keep_aspect_ratio=True,
        cover_mode=cover_mode
    )
    
    if scale_factor != 0.0:
        _, overlay_h, overlay_w, _ = resized_overlay.shape
        scale_multiplier = 1.0 + (scale_factor / 100.0)
        new_w = max(1, int(overlay_w * scale_multiplier))
        new_h = max(1, int(overlay_h * scale_multiplier))
        img_np = resized_overlay[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        resized_overlay = torch.from_numpy(img_np)[None,]
    
    _, overlay_h, overlay_w, _ = resized_overlay.shape
    
    mask_np = overlay_mask[0].cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)
    mask_pil = mask_pil.resize((overlay_w, overlay_h), Image.LANCZOS)
    
    if feather > 0:
        from PIL import ImageFilter
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather))
    
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    resized_mask = torch.from_numpy(mask_np)[None,]
    
    result = base_image.clone()
    
    if cover_mode and allow_crop and (overlay_w > target_width or overlay_h > target_height):
        if alignment == "top":
            crop_top = 0
            crop_left = max(0, (overlay_w - target_width) // 2)
        elif alignment == "bottom":
            crop_top = max(0, overlay_h - target_height)
            crop_left = max(0, (overlay_w - target_width) // 2)
        elif alignment == "left":
            crop_left = 0
            crop_top = max(0, (overlay_h - target_height) // 2)
        elif alignment == "right":
            crop_left = max(0, overlay_w - target_width)
            crop_top = max(0, (overlay_h - target_height) // 2)
        else:
            crop_left = max(0, (overlay_w - target_width) // 2)
            crop_top = max(0, (overlay_h - target_height) // 2)
        
        crop_right = crop_left + min(overlay_w - crop_left, target_width)
        crop_bottom = crop_top + min(overlay_h - crop_top, target_height)
        
        resized_overlay = resized_overlay[:, crop_top:crop_bottom, crop_left:crop_right, :]
        resized_mask = resized_mask[:, crop_top:crop_bottom, crop_left:crop_right]
        
        _, overlay_h, overlay_w, _ = resized_overlay.shape
        
        paste_left = left
        paste_top = top
    else:
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
        else:
            paste_left = left + (target_width - overlay_w) // 2
            paste_top = top + (target_height - overlay_h) // 2
    
    paste_left = paste_left + offset_x
    paste_top = paste_top + offset_y
    
    mask_3ch = resized_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
    
    src_start_x = max(0, -paste_left)
    src_start_y = max(0, -paste_top)
    dst_start_x = max(0, paste_left)
    dst_start_y = max(0, paste_top)
    
    paste_w = min(overlay_w - src_start_x, base_image.shape[2] - dst_start_x)
    paste_h = min(overlay_h - src_start_y, base_image.shape[1] - dst_start_y)
    
    if paste_w <= 0 or paste_h <= 0:
        return result
    
    final_paste_left = dst_start_x
    final_paste_top = dst_start_y
    
    result[
        :,
        final_paste_top:final_paste_top+paste_h,
        final_paste_left:final_paste_left+paste_w,
        :
    ] = (
        base_image[
            :,
            final_paste_top:final_paste_top+paste_h,
            final_paste_left:final_paste_left+paste_w,
            :
        ] * (1 - mask_3ch[:, src_start_y:src_start_y+paste_h, src_start_x:src_start_x+paste_w, :]) +
        resized_overlay[:, src_start_y:src_start_y+paste_h, src_start_x:src_start_x+paste_w, :] * mask_3ch[:, src_start_y:src_start_y+paste_h, src_start_x:src_start_x+paste_w, :]
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
    根据遮罩智能替换图片物体（基础版，保持与bak一致）
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
        # 获取基础图片遮罩的边界框
        target_bbox = get_mask_bounding_box(base_mask)
        
        if replace_mask is None:
            _, h, w, _ = replace_image.shape
            replace_mask = torch.ones(1, h, w)
        
        if replace_mask.dim() == 2:
            replace_mask = replace_mask.unsqueeze(0)
        
        cropped_replace = replace_image
        cropped_mask = replace_mask
        
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


class ImageReplaceWithMaskV2:
    """
    根据遮罩智能替换图片物体（增强版，含偏移/缩放/裁切）
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
                "offset_left": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999
                }),
                "offset_right": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999
                }),
                "offset_up": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999
                }),
                "offset_down": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999
                }),
                "allow_crop": ("BOOLEAN", {
                    "default": True,
                    "label_on": "允许裁切",
                    "label_off": "禁用裁切"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 0.0,
                    "min": -99.0,
                    "max": 9999.0,
                    "step": 0.1
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
        offset_left,
        offset_right,
        offset_up,
        offset_down,
        allow_crop,
        scale_factor,
        replace_mask=None
    ):
        # 获取基础图片遮罩的边界框
        target_bbox = get_mask_bounding_box(base_mask)
        
        if replace_mask is None:
            _, h, w, _ = replace_image.shape
            replace_mask = torch.ones(1, h, w)
        
        if replace_mask.dim() == 2:
            replace_mask = replace_mask.unsqueeze(0)
        
        cropped_replace = replace_image
        cropped_mask = replace_mask
        
        offset_x = offset_right - offset_left
        offset_y = offset_down - offset_up
        
        result = composite_images_v2(
            base_image,
            cropped_replace,
            cropped_mask,
            target_bbox,
            feather,
            cover_mode=cover_mode,
            alignment=alignment,
            offset_x=offset_x,
            offset_y=offset_y,
            allow_crop=allow_crop,
            scale_factor=scale_factor
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


class ReplaceBackgroundWithWhiteExpand:
    """替换背景为白色，并可扩展空白区域以缩小遮罩物品占比"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "expand_up": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999,
                    "tooltip": "向上扩展空白区域（正数扩展空白缩小物品，负数反向扩展）"
                }),
                "expand_down": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999,
                    "tooltip": "向下扩展空白区域（正数扩展空白缩小物品，负数反向扩展）"
                }),
                "expand_left": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999,
                    "tooltip": "向左扩展空白区域（正数扩展空白缩小物品，负数反向扩展）"
                }),
                "expand_right": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999,
                    "tooltip": "向右扩展空白区域（正数扩展空白缩小物品，负数反向扩展）"
                }),
                "background_alpha": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "背景透明度 (0.0=完全白色, 1.0=保持原图)"
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    def main(self, image, expand_up, expand_down, expand_left, expand_right, background_alpha, mask=None):
        """
        替换背景为白色，并向外扩展画布尺寸以缩小遮罩物品占比
        
        Args:
            image: 输入图片
            expand_up: 向上扩展画布（正数向上扩展，负数反向扩展）
            expand_down: 向下扩展画布（正数向下扩展，负数反向扩展）
            expand_left: 向左扩展画布（正数向左扩展，负数反向扩展）
            expand_right: 向右扩展画布（正数向右扩展，负数反向扩展）
            background_alpha: 背景透明度 (0.0=完全白色, 1.0=保持原图)
            mask: 可选输入遮罩，如果提供则根据遮罩处理背景
            
        Returns:
            扩展后的图片（新尺寸）
        """
        # 获取原图片尺寸
        _, img_h, img_w, _ = image.shape
        
        # 计算新画布尺寸（向外扩展）
        new_width = img_w + expand_left + expand_right
        new_height = img_h + expand_up + expand_down
        
        # 确保新尺寸至少为1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # 创建新的白色画布
        white_background = torch.ones(1, new_height, new_width, 3, device=image.device, dtype=image.dtype)
        
        # 计算原图片在新画布中的位置
        # 如果expand_left为正，原图向右移动；如果为负，原图向左移动（需要裁剪左侧）
        paste_x = max(0, expand_left)  # 原图在新画布中的x位置
        paste_y = max(0, expand_up)    # 原图在新画布中的y位置
        
        # 计算原图中要粘贴的区域
        # 如果expand_left为负，需要裁剪原图左侧
        src_start_x = max(0, -expand_left)   # 原图左侧裁剪量
        src_start_y = max(0, -expand_up)     # 原图上方裁剪量
        # 如果expand_right为负，需要裁剪原图右侧
        src_end_x = img_w - max(0, -expand_right)   # 原图右侧保留到此处
        src_end_y = img_h - max(0, -expand_down)     # 原图下方保留到此处
        
        # 确保有效区域
        src_end_x = max(src_start_x, src_end_x)
        src_end_y = max(src_start_y, src_end_y)
        
        # 计算粘贴尺寸
        paste_width = src_end_x - src_start_x
        paste_height = src_end_y - src_start_y
        
        # 计算目标粘贴区域
        dst_start_x = paste_x
        dst_start_y = paste_y
        dst_end_x = paste_x + paste_width
        dst_end_y = paste_y + paste_height
        
        # 将原图片粘贴到新画布上
        result_image = white_background.clone()
        if paste_width > 0 and paste_height > 0:
            result_image[
                :,
                dst_start_y:dst_end_y,
                dst_start_x:dst_end_x,
                :
            ] = image[
                :,
                src_start_y:src_end_y,
                src_start_x:src_end_x,
                :
            ]
        
        # 如果没有提供mask，直接返回扩展后的图片（背景已经是白色）
        if mask is None:
            return (result_image,)
        
        # 如果有mask，处理遮罩逻辑
        # 确保 mask 是 3D 的 (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # 创建扩展后的遮罩
        expanded_mask = torch.zeros(1, new_height, new_width, device=mask.device, dtype=mask.dtype)
        
        # 将原遮罩粘贴到新位置（使用相同的粘贴区域）
        if paste_width > 0 and paste_height > 0:
            expanded_mask[
                :,
                dst_start_y:dst_end_y,
                dst_start_x:dst_end_x
            ] = mask[
                :,
                src_start_y:src_end_y,
                src_start_x:src_end_x
            ]
        
        # 扩展 mask 维度以匹配 RGB 通道
        mask_3ch = expanded_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        
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


class FillMaskWithColor:
    """根据遮罩将图片的遮罩区域填充成指定颜色"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "R": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "G": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "B": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
            }
        }
    
    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    def main(self, image, mask, R, G, B):
        """
        将遮罩区域填充成指定颜色
        
        Args:
            image: 输入图片
            mask: 输入遮罩（遮罩区域将被填充）
            R: 红色通道值 (0-255)
            G: 绿色通道值 (0-255)
            B: 蓝色通道值 (0-255)
            
        Returns:
            处理后的图片（遮罩区域填充为指定颜色）
        """
        # 确保 mask 是 3D 的 (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # 确保遮罩和图片尺寸匹配
        _, img_h, img_w, _ = image.shape
        _, mask_h, mask_w = mask.shape
        
        # 如果尺寸不匹配，需要调整遮罩大小
        if mask_h != img_h or mask_w != img_w:
            # 转换遮罩为PIL并调整大小
            mask_np = mask[0].cpu().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
            mask_pil = mask_pil.resize((img_w, img_h), Image.LANCZOS)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask_np)[None,]
        
        # 克隆原图
        result_image = image.clone()
        
        # 将RGB值从0-255转换为0-1范围
        fill_color = torch.tensor([
            R / 255.0,
            G / 255.0,
            B / 255.0
        ], device=image.device, dtype=image.dtype)
        
        # 扩展遮罩维度以匹配RGB通道: (1, H, W) -> (1, H, W, 3)
        mask_3ch = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        
        # 扩展颜色张量以匹配图片尺寸: (3,) -> (1, H, W, 3)
        fill_color_3d = fill_color.view(1, 1, 1, 3).expand_as(result_image)
        
        # 在遮罩区域填充颜色：
        # - 遮罩区域 (mask≈1): 使用指定颜色
        # - 非遮罩区域 (mask≈0): 保持原图
        result_image = (
            result_image * (1.0 - mask_3ch) +  # 非遮罩区域：保持原图
            fill_color_3d * mask_3ch  # 遮罩区域：填充指定颜色
        )
        
        return (result_image,)


class MergeMasks:
    """合并多个遮罩为一个遮罩"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }
    
    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("merged_mask",)
    
    def main(self, masks):
        """
        合并多个遮罩为一个遮罩
        
        Args:
            masks: 批量遮罩，shape为 (N, H, W) 或 (N, 1, H, W)
            
        Returns:
            合并后的遮罩 shape (1, H, W)
        """
        # 确保masks是tensor
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)
        
        # 处理不同的输入格式
        if masks.dim() == 2:
            # (H, W) -> (1, H, W)
            masks = masks.unsqueeze(0)
        elif masks.dim() == 4:
            # (N, 1, H, W) -> (N, H, W)
            masks = masks.squeeze(1)
        
        # 现在masks应该是 (N, H, W) 格式
        if masks.dim() != 3:
            raise ValueError(f"遮罩格式不正确，期望 (N, H, W)，得到 {masks.shape}")
        
        # 合并遮罩：对所有遮罩取最大值（取并集）
        # 这样可以保留所有物体的遮罩区域
        merged_mask = torch.max(masks, dim=0)[0]  # 对第一个维度取最大值
        
        # 确保输出格式为 (1, H, W)
        if merged_mask.dim() == 2:
            merged_mask = merged_mask.unsqueeze(0)
        
        return (merged_mask,)


class SelectLargestMask:
    """根据boxes面积筛选出最大的遮罩"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "boxes": ("STRING", {
                    "default": "[]",
                    "multiline": True
                }),
            }
        }
    
    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("largest_mask", "index")
    
    def main(self, masks, boxes):
        """
        根据boxes面积筛选出最大的遮罩
        
        Args:
            masks: 批量遮罩，shape为 (N, H, W) 或 (N, 1, H, W)
            boxes: boxes字符串或列表，格式如 "[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]"
                  或 [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            
        Returns:
            largest_mask: 最大的遮罩 shape (1, H, W)
            index: 最大遮罩的索引
        """
        # 确保masks是tensor
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)
        
        # 处理不同的输入格式
        if masks.dim() == 2:
            # (H, W) -> (1, H, W)
            masks = masks.unsqueeze(0)
        elif masks.dim() == 4:
            # (N, 1, H, W) -> (N, H, W)
            masks = masks.squeeze(1)
        
        # 现在masks应该是 (N, H, W) 格式
        if masks.dim() != 3:
            raise ValueError(f"遮罩格式不正确，期望 (N, H, W)，得到 {masks.shape}")
        
        num_masks = masks.shape[0]
        
        # 解析boxes
        boxes_list = None
        try:
            if isinstance(boxes, str):
                # 尝试解析JSON字符串
                try:
                    boxes_list = json.loads(boxes)
                except:
                    # JSON解析失败，尝试作为Python字面量（更安全的方式）
                    try:
                        # 使用ast.literal_eval替代eval，更安全
                        boxes_list = ast.literal_eval(boxes)
                    except:
                        boxes_list = None
            elif isinstance(boxes, (list, tuple)):
                boxes_list = boxes
            elif isinstance(boxes, torch.Tensor):
                # 如果是tensor，转换为列表
                boxes_list = boxes.cpu().tolist()
            
            # 处理嵌套列表格式：[[[x1,y1,x2,y2], ...], ...] -> [[x1,y1,x2,y2], ...]
            if boxes_list and len(boxes_list) > 0:
                if isinstance(boxes_list[0], list) and len(boxes_list[0]) > 0:
                    if isinstance(boxes_list[0][0], list) or isinstance(boxes_list[0][0], (int, float)):
                        # 检查是否是嵌套格式
                        first_item = boxes_list[0]
                        if isinstance(first_item, list) and len(first_item) > 0:
                            if isinstance(first_item[0], list):
                                # 是嵌套格式 [[[x1,y1,x2,y2], ...], ...]，取第一层
                                boxes_list = boxes_list[0] if boxes_list else []
            
        except Exception as e:
            boxes_list = None
        
        # 计算每个box的面积
        areas = []
        if boxes_list and len(boxes_list) > 0:
            for box in boxes_list:
                try:
                    # 处理不同的box格式
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        # [x1, y1, x2, y2] 格式
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        area = abs((x2 - x1) * (y2 - y1))
                        areas.append(area)
                    elif isinstance(box, torch.Tensor) and box.numel() >= 4:
                        # tensor格式
                        box_values = box.cpu().tolist()
                        if isinstance(box_values, list):
                            x1, y1, x2, y2 = float(box_values[0]), float(box_values[1]), float(box_values[2]), float(box_values[3])
                        else:
                            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        area = abs((x2 - x1) * (y2 - y1))
                        areas.append(area)
                    else:
                        areas.append(0)
                except:
                    areas.append(0)
        
        # 如果没有有效的boxes，或者boxes数量与masks不匹配，使用遮罩像素数
        if not areas or len(areas) != num_masks:
            # 计算每个遮罩的非零像素数（面积）
            areas = []
            for i in range(num_masks):
                mask_area = torch.sum(masks[i] > 0).item()
                areas.append(mask_area)
        
        # 找到面积最大的索引
        if not areas:
            # 如果没有有效的面积数据，返回第一个遮罩
            largest_idx = 0
        else:
            largest_idx = int(np.argmax(areas))
        
        # 提取最大的遮罩
        largest_mask = masks[largest_idx].unsqueeze(0)  # (H, W) -> (1, H, W)
        
        return (largest_mask, largest_idx)


class SelectLargestMaskByArea:
    """根据mask面积直接筛选出最大的遮罩（不依赖bbox）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }
    
    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("largest_mask", "index")
    
    def main(self, masks):
        """
        按连通域面积筛选最大的遮罩（阈值二值化 + 最大连通域，类似 Mask-filter）
        
        Args:
            masks: 批量遮罩，shape为 (N, H, W) 或 (N, 1, H, W) 或 (H, W)
        """
        # 确保masks是tensor
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)
        
        # 处理不同的输入格式
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)          # (H, W) -> (1, H, W)
        elif masks.dim() == 4:
            masks = masks.squeeze(1)            # (N, 1, H, W) -> (N, H, W)
        
        if masks.dim() != 3:
            raise ValueError(f"遮罩格式不正确，期望 (N, H, W)，得到 {masks.shape}")
        
        # 若只有单个遮罩，直接用连通域筛一遍
        if masks.shape[0] == 1:
            largest_mask = self._largest_component(masks[0])
            return (largest_mask.unsqueeze(0), 0)
        
        # 遍历批次，取每个mask的最大连通域面积，再整体取最大
        best_idx = 0
        best_area = -1
        best_mask = None
        for i in range(masks.shape[0]):
            lm = self._largest_component(masks[i])
            area = lm.sum().item()
            if area > best_area:
                best_area = area
                best_idx = i
                best_mask = lm
        
        if best_mask is None:
            # fallback：返回第一个
            return (masks[0].unsqueeze(0), 0)
        
        return (best_mask.unsqueeze(0), best_idx)
    
    def _largest_component(self, mask_tensor, threshold: float = 0.5):
        """
        对单个遮罩取最大连通域，返回 (H, W) tensor
        """
        mask_np = mask_tensor.cpu().numpy().astype(np.float32)
        if mask_np.ndim > 2:
            mask_np = mask_np[0]
        
        # 二值化
        binary = (mask_np >= threshold).astype(np.uint8)
        
        # 找连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return torch.from_numpy(binary.astype(np.float32))
        
        # 最大前景（排除背景label 0）
        max_label = 1
        max_area = stats[1, cv2.CC_STAT_AREA]
        for lbl in range(2, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = lbl
        
        largest = (labels == max_label).astype(np.float32)
        return torch.from_numpy(largest)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskBoundingBox": MaskBoundingBox,
    "CropImageByMask": CropImageByMask,
    "ImageReplaceWithMask": ImageReplaceWithMask,
    "ImageReplaceWithMaskV2": ImageReplaceWithMaskV2,
    "CropImageWithWhiteBackground": CropImageWithWhiteBackground,
    "ReplaceBackgroundWithWhite": ReplaceBackgroundWithWhite,
    "ReplaceBackgroundWithWhiteExpand": ReplaceBackgroundWithWhiteExpand,
    "VisualizeDetectionBox": VisualizeDetectionBox,
    "FillMaskWithColor": FillMaskWithColor,
    "MergeMasks": MergeMasks,
    "SelectLargestMask": SelectLargestMask,
    "SelectLargestMaskByArea": SelectLargestMaskByArea,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBoundingBox": "提取遮罩边界框",
    "CropImageByMask": "按遮罩裁剪图片",
    "ImageReplaceWithMask": "智能物体替换",
    "ImageReplaceWithMaskV2": "智能物体替换 V2",
    "CropImageWithWhiteBackground": "裁剪图片并替换背景为白色",
    "ReplaceBackgroundWithWhite": "只替换背景为白色",
    "ReplaceBackgroundWithWhiteExpand": "替换背景为白色（可扩展空白）",
    "VisualizeDetectionBox": "可视化检测框",
    "FillMaskWithColor": "遮罩区域填充颜色",
    "MergeMasks": "合并遮罩",
    "SelectLargestMask": "筛选最大遮罩",
    "SelectLargestMaskByArea": "筛选最大遮罩（按面积）",
}






