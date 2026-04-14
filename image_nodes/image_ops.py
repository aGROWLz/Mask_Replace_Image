import torch
import numpy as np
from PIL import Image, ImageDraw
from ..mask_nodes.utils import get_mask_bounding_box, crop_image_by_mask, resize_image_to_fit
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

class CropImageWithWhiteBackground:
    """根据遮罩裁剪图片并将背景替换为白色，支持背景透明度控制"""

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
        根据遮罩裁剪图片并处理背景透明度

        Args:
            image: 输入图片
            mask: 输入遮罩
            background_alpha: 背景透明度 (0.0=完全不透明白色, 1.0=完全透明)

        Returns:
            处理后的图片(RGBA格式)和裁剪后的遮罩
        """
        # 确保 mask 是 3D 的 (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # 获取边界框
        left, top, right, bottom = get_mask_bounding_box(mask)

        if left == right or top == bottom:
            # 如果边界框无效，返回原图(转换为RGBA)
            b, h, w, c = image.shape
            if c == 3:
                alpha = torch.ones(b, h, w, 1, device=image.device, dtype=image.dtype)
                image_rgba = torch.cat([image, alpha], dim=-1)
            else:
                image_rgba = image
            return (image_rgba, mask)

        # 裁剪图片和遮罩
        cropped_image = image[:, top:bottom+1, left:right+1, :].clone()
        cropped_mask = mask[:, top:bottom+1, left:right+1]

        b, h, w, c = cropped_image.shape

        # 创建白色背景 (RGB)
        white_rgb = torch.ones(b, h, w, 3, device=cropped_image.device, dtype=cropped_image.dtype)

        # 确保图像是RGB格式
        if c == 4:
            cropped_rgb = cropped_image[:, :, :, :3]
        else:
            cropped_rgb = cropped_image

        # 扩展 mask 到 3 通道用于RGB混合
        mask_3ch = cropped_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        # RGB部分：物体保持原图，背景使用白色
        result_rgb = cropped_rgb * mask_3ch + white_rgb * (1.0 - mask_3ch)

        # Alpha通道：物体不透明，背景根据background_alpha参数控制透明度
        # mask=1(物体): alpha=1(不透明)
        # mask=0(背景): alpha=(1-background_alpha)  注意：颠倒后的逻辑
        # background_alpha=0 -> alpha=1 (不透明/白色)
        # background_alpha=1 -> alpha=0 (透明)
        bg_alpha = 1.0 - background_alpha
        alpha_channel = cropped_mask * 1.0 + (1.0 - cropped_mask) * bg_alpha
        alpha_channel = alpha_channel.unsqueeze(-1)  # (B, H, W, 1)

        # 合并RGB和Alpha通道
        result_rgba = torch.cat([result_rgb, alpha_channel], dim=-1)

        return (result_rgba, cropped_mask)

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
    """替换背景为白色，并可扩展空白区域以缩小遮罩物品占比，支持透明背景"""

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
                    "tooltip": "背景透明度 (0.0=完全不透明白色, 1.0=完全透明)"
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
            image: 输入图片 (支持RGB或RGBA)
            expand_up: 向上扩展画布（正数向上扩展，负数反向扩展）
            expand_down: 向下扩展画布（正数向下扩展，负数反向扩展）
            expand_left: 向左扩展画布（正数向左扩展，负数反向扩展）
            expand_right: 向右扩展画布（正数向右扩展，负数反向扩展）
            background_alpha: 背景透明度 (0.0=完全不透明白色, 1.0=完全透明)
            mask: 可选输入遮罩，如果提供则根据遮罩处理背景

        Returns:
            扩展后的图片（RGBA格式，新尺寸）
        """
        # 获取原图片尺寸和通道数
        _, img_h, img_w, img_c = image.shape

        # 分离RGB和Alpha通道（如果有）
        if img_c == 4:
            image_rgb = image[:, :, :, :3]
            image_alpha = image[:, :, :, 3:4]
        else:
            image_rgb = image
            image_alpha = torch.ones(1, img_h, img_w, 1, device=image.device, dtype=image.dtype)

        # 计算新画布尺寸（向外扩展）
        new_width = img_w + expand_left + expand_right
        new_height = img_h + expand_up + expand_down

        # 确保新尺寸至少为1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # 创建新的白色画布 (RGB)
        white_rgb = torch.ones(1, new_height, new_width, 3, device=image.device, dtype=image.dtype)
        # 创建透明Alpha画布
        transparent_alpha = torch.zeros(1, new_height, new_width, 1, device=image.device, dtype=image.dtype)

        # 计算原图片在新画布中的位置
        paste_x = max(0, expand_left)
        paste_y = max(0, expand_up)

        # 计算原图中要粘贴的区域
        src_start_x = max(0, -expand_left)
        src_start_y = max(0, -expand_up)
        src_end_x = img_w - max(0, -expand_right)
        src_end_y = img_h - max(0, -expand_down)

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

        # 将原图片RGB粘贴到新画布上
        result_rgb = white_rgb.clone()
        # 将原图片Alpha粘贴到新画布上
        result_alpha = transparent_alpha.clone()

        if paste_width > 0 and paste_height > 0:
            result_rgb[
                :,
                dst_start_y:dst_end_y,
                dst_start_x:dst_end_x,
                :
            ] = image_rgb[
                :,
                src_start_y:src_end_y,
                src_start_x:src_end_x,
                :
            ]
            result_alpha[
                :,
                dst_start_y:dst_end_y,
                dst_start_x:dst_end_x,
                :
            ] = image_alpha[
                :,
                src_start_y:src_end_y,
                src_start_x:src_end_x,
                :
            ]

        # 转换透明度：background_alpha=0(不透明) -> bg_alpha=1, background_alpha=1(透明) -> bg_alpha=0
        bg_alpha = 1.0 - background_alpha

        # 如果没有提供mask，直接返回扩展后的图片（背景根据background_alpha控制透明度）
        if mask is None:
            # 背景区域（新扩展的部分）使用background_alpha
            # 创建一个遮罩表示原图区域
            original_mask = torch.zeros(1, new_height, new_width, 1, device=image.device, dtype=image.dtype)
            if paste_width > 0 and paste_height > 0:
                original_mask[
                    :,
                    dst_start_y:dst_end_y,
                    dst_start_x:dst_end_x,
                    :
                ] = 1.0

            # Alpha通道：原图区域保持原alpha，新扩展区域使用bg_alpha
            final_alpha = result_alpha * original_mask + bg_alpha * (1.0 - original_mask)
            result_rgba = torch.cat([result_rgb, final_alpha], dim=-1)
            return (result_rgba,)

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

        # 扩展 mask 到 4 通道用于RGBA混合
        mask_3ch = expanded_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        mask_1ch = expanded_mask.unsqueeze(-1)  # (B, H, W, 1)

        # RGB部分：物体区域保持原图，背景区域使用白色
        result_rgb = result_rgb * mask_3ch + white_rgb * (1.0 - mask_3ch)

        # Alpha通道：
        # - 物体区域 (mask=1): 保持原alpha值（不透明）
        # - 背景区域 (mask=0): 使用bg_alpha参数控制透明度
        final_alpha = mask_1ch * 1.0 + (1.0 - mask_1ch) * bg_alpha

        # 合并RGB和Alpha通道
        result_rgba = torch.cat([result_rgb, final_alpha], dim=-1)

        return (result_rgba,)

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

