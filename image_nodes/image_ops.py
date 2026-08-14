import torch
import numpy as np
import cv2
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


class CropImageWithPosition:
    """根据遮罩裁剪图像，支持四方向调整，输出裁剪信息和原图"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "expand_up": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999,
                    "tooltip": "向上扩展裁剪区域（像素）"
                }),
                "expand_down": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999,
                    "tooltip": "向下扩展裁剪区域（像素）"
                }),
                "expand_left": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999,
                    "tooltip": "向左扩展裁剪区域（像素）"
                }),
                "expand_right": ("INT", {
                    "default": 0,
                    "min": -99999,
                    "max": 99999,
                    "tooltip": "向右扩展裁剪区域（像素）"
                }),
                "force_square": ("BOOLEAN", {
                    "default": False,
                    "label_on": "强制1:1比例",
                    "label_off": "保持原比例",
                    "tooltip": "开启后会根据expand参数自动调整为正方形裁剪区域"
                }),
            }
        }

    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "original_image", "crop_position")

    def main(self, image, mask, expand_up, expand_down, expand_left, expand_right, force_square):
        """
        根据遮罩裁剪图像，支持四方向调整

        Args:
            image: 输入图像
            mask: 输入遮罩
            expand_up: 向上扩展像素数
            expand_down: 向下扩展像素数
            expand_left: 向左扩展像素数
            expand_right: 向右扩展像素数
            force_square: 是否强制1:1比例

        Returns:
            cropped_image: 裁剪后的图像
            cropped_mask: 裁剪后的遮罩（用于后续处理）
            original_image: 原图（用于后续贴回）
            crop_position: 裁剪位置信息（JSON格式）
        """
        import json
        from ..mask_nodes.utils import get_mask_bounding_box

        # 确保 mask 是 3D 的
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # 获取遮罩边界框
        left, top, right, bottom = get_mask_bounding_box(mask)

        if left == right or top == bottom:
            # 遮罩为空，返回原图
            position_info = json.dumps({
                "left": 0,
                "top": 0,
                "right": image.shape[2] - 1,
                "bottom": image.shape[1] - 1,
                "original_width": image.shape[2],
                "original_height": image.shape[1],
                "is_empty": True
            })
            # 返回全0遮罩
            empty_mask = torch.zeros(1, image.shape[1], image.shape[2], device=image.device, dtype=torch.float32)
            return (image.clone(), empty_mask, image.clone(), position_info)

        # 应用扩展
        left = max(0, left - expand_left)
        top = max(0, top - expand_up)
        right = min(image.shape[2] - 1, right + expand_right)
        bottom = min(image.shape[1] - 1, bottom + expand_down)

        # 如果强制1:1比例，调整裁剪区域
        if force_square:
            # 计算当前裁剪区域的中心点
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            # 计算当前宽高
            current_width = right - left + 1
            current_height = bottom - top + 1

            # 取较大的一边作为正方形边长
            square_size = max(current_width, current_height)

            # 以中心点为基准，计算新的正方形边界
            half_size = square_size // 2
            new_left = center_x - half_size
            new_right = center_x + half_size
            new_top = center_y - half_size
            new_bottom = center_y + half_size

            # 确保不超出原图边界
            if new_left < 0:
                new_right -= new_left
                new_left = 0
            if new_top < 0:
                new_bottom -= new_top
                new_top = 0
            if new_right >= image.shape[2]:
                new_left -= (new_right - image.shape[2] + 1)
                new_right = image.shape[2] - 1
            if new_bottom >= image.shape[1]:
                new_top -= (new_bottom - image.shape[1] + 1)
                new_bottom = image.shape[1] - 1

            # 再次确保不越界
            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(image.shape[2] - 1, new_right)
            new_bottom = min(image.shape[1] - 1, new_bottom)

            # 更新裁剪区域
            left, top, right, bottom = new_left, new_top, new_right, new_bottom

        # 裁剪图像和遮罩
        cropped = image[:, top:bottom+1, left:right+1, :].clone()
        cropped_mask = mask[:, top:bottom+1, left:right+1].clone()

        # 保存位置信息
        position_info = json.dumps({
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "original_width": image.shape[2],
            "original_height": image.shape[1],
            "cropped_width": cropped.shape[2],
            "cropped_height": cropped.shape[1],
            "is_empty": False,
            "force_square": force_square
        })

        return (cropped, cropped_mask, image.clone(), position_info)


class PasteCroppedImage:
    """将处理后的裁剪图像贴回原图"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "crop_position": ("STRING", {
                    "multiline": True,
                    "tooltip": "裁剪位置信息（JSON格式）"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "边缘羽化像素数"
                }),
            }
        }

    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    def main(self, processed_image, original_image, crop_position, feather):
        """
        将处理后的裁剪图像贴回原图

        Args:
            processed_image: 处理后的裁剪图像（比例相同，分辨率可能不同）
            original_image: 原图
            crop_position: 裁剪位置信息（JSON格式）
            feather: 边缘羽化像素数

        Returns:
            贴回后的完整图像
        """
        import json
        from PIL import Image, ImageFilter

        # 解析位置信息
        try:
            pos = json.loads(crop_position)
        except:
            raise ValueError("crop_position 必须是有效的 JSON 字符串")

        if pos.get("is_empty", False):
            # 如果原裁剪区域为空，直接返回原图
            return (original_image,)

        left = pos["left"]
        top = pos["top"]
        right = pos["right"]
        bottom = pos["bottom"]
        original_width = pos["original_width"]
        original_height = pos["original_height"]

        # 目标区域的尺寸
        target_width = right - left + 1
        target_height = bottom - top + 1

        # 处理后的图像尺寸
        proc_h, proc_w = processed_image.shape[1], processed_image.shape[2]

        # 将处理后的图像缩放到目标区域大小
        if proc_h != target_height or proc_w != target_width:
            # 转换为PIL进行缩放
            proc_np = processed_image[0].cpu().numpy()
            proc_np = (proc_np * 255).astype(np.uint8)
            proc_pil = Image.fromarray(proc_np)
            proc_pil = proc_pil.resize((target_width, target_height), Image.LANCZOS)
            proc_np = np.array(proc_pil).astype(np.float32) / 255.0
            processed_image = torch.from_numpy(proc_np)[None,].to(processed_image.device, processed_image.dtype)

        # 创建输出图像（复制原图）
        result = original_image.clone()

        # 创建遮罩（用于羽化）
        mask = torch.ones(1, target_height, target_width, device=original_image.device, dtype=torch.float32)

        # 应用羽化
        if feather > 0:
            mask_np = mask[0].cpu().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather))
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask_np)[None,].to(original_image.device, torch.float32)

        # 扩展到3通道
        mask_3ch = mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        # 贴回图像
        result[:, top:bottom+1, left:right+1, :] = (
            original_image[:, top:bottom+1, left:right+1, :] * (1 - mask_3ch) +
            processed_image * mask_3ch
        )

        return (result,)


class MaskToCropPosition:
    """将遮罩转换为裁剪位置信息（crop_position JSON 字符串）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "image_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "原图宽度；为 0 时使用遮罩自身宽度"
                }),
                "image_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "原图高度；为 0 时使用遮罩自身高度"
                }),
            }
        }

    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("crop_position",)

    def main(self, mask, image_width, image_height):
        """
        将遮罩转换为 crop_position（JSON 字符串）

        Args:
            mask: 输入遮罩
            image_width: 原图宽度（0 则用遮罩宽度）
            image_height: 原图高度（0 则用遮罩高度）

        Returns:
            crop_position JSON 字符串
        """
        import json

        if mask.dim() == 3:
            mask = mask.squeeze(0)

        mask_np = mask.cpu().numpy()
        rows = np.any(mask_np > 0, axis=1)
        cols = np.any(mask_np > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            w = int(image_width) if image_width > 0 else int(mask_np.shape[1])
            h = int(image_height) if image_height > 0 else int(mask_np.shape[0])
            pos = {
                "left": 0,
                "top": 0,
                "right": w - 1,
                "bottom": h - 1,
                "original_width": w,
                "original_height": h,
                "is_empty": True,
            }
            return (json.dumps(pos),)

        top = int(np.argmax(rows))
        bottom = int(len(rows) - np.argmax(rows[::-1]) - 1)
        left = int(np.argmax(cols))
        right = int(len(cols) - np.argmax(cols[::-1]) - 1)

        w = int(image_width) if image_width > 0 else int(mask_np.shape[1])
        h = int(image_height) if image_height > 0 else int(mask_np.shape[0])

        pos = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "original_width": w,
            "original_height": h,
            "is_empty": False,
        }

        return (json.dumps(pos),)


class MaskEdgeMarker:
    """沿遮罩边缘生成标记遮罩和红色标记图层（无贴回功能）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "expand_outward": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "标记区域向外扩展（像素），覆盖遮罩外侧"
                }),
                "expand_inward": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "标记区域向内扩展（像素），覆盖遮罩内侧"
                }),
                "marker_alpha": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "红色标记图层透明度"
                }),
                "mask_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "边缘遮罩透明度（用于生图模型修复）"
                }),
            }
        }

    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("marked_image", "edge_mask")

    def main(self, image, mask, expand_outward, expand_inward,
             marker_alpha, mask_alpha):
        """
        沿遮罩边缘生成红色标记图层和边缘遮罩

        Args:
            image: 输入图像
            mask: 输入遮罩（标记沿其边缘生成）
            expand_outward: 标记区域向外扩展像素
            expand_inward: 标记区域向内扩展像素
            marker_alpha: 红色标记图层透明度
            mask_alpha: 边缘遮罩透明度

        Returns:
            marked_image: 带红色边缘标记的图像（RGBA）
            edge_mask: 边缘区域遮罩
        """
        img_h, img_w = image.shape[1], image.shape[2]

        # --- 1. 归一化遮罩为 (H, W) uint8 二值图 ---
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        mask_np = mask.cpu().numpy()
        if mask_np.ndim > 2:
            mask_np = mask_np[0]
        binary = (mask_np > 0.5).astype(np.uint8)

        if not np.any(binary):
            empty_mask = torch.zeros(1, img_h, img_w,
                                     device=image.device, dtype=torch.float32)
            # 无有效遮罩时，返回原图（转RGBA）和空遮罩
            b, h, w, c = image.shape
            if c == 3:
                alpha = torch.ones(b, h, w, 1, device=image.device, dtype=image.dtype)
                image_rgba = torch.cat([image, alpha], dim=-1)
            else:
                image_rgba = image
            return (image_rgba, empty_mask)

        # --- 2. 沿 mask 真实轮廓计算边缘环带（任意形状）---
        # 外扩：膨胀 expand_outward 像素
        dilated = binary
        if expand_outward > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (expand_outward * 2 + 1, expand_outward * 2 + 1))
            dilated = cv2.dilate(binary, k, iterations=1)

        # 内缩：腐蚀 expand_inward 像素
        eroded = binary
        if expand_inward > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (expand_inward * 2 + 1, expand_inward * 2 + 1))
            eroded = cv2.erode(binary, k, iterations=1)

        # 边缘环带 = 外扩区域 - 内缩区域
        edge_np = np.clip(dilated.astype(np.int16) - eroded.astype(np.int16), 0, 1).astype(np.float32)
        edge_mask = torch.from_numpy(edge_np)[None,].to(image.device, torch.float32)

        # --- 3. 创建带红色标记的输出图像 ---
        result_rgb = image[:, :, :, :3].clone()  # 只取RGB
        _, h, w, _ = result_rgb.shape

        # 红色标记图层: (1, H, W, 3)，值为 (1, 0, 0)
        red_overlay = torch.zeros(1, h, w, 3, device=image.device, dtype=image.dtype)
        red_overlay[:, :, :, 0] = 1.0  # R通道

        # 扩展 edge_mask 到 3 通道
        edge_mask_3ch = edge_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        # 混合: result * (1 - edge_mask * marker_alpha) + red * edge_mask * marker_alpha
        marked_rgb = result_rgb * (1.0 - edge_mask_3ch * marker_alpha) \
                     + red_overlay * edge_mask_3ch * marker_alpha

        # 转为 RGBA，红色标记区域半透明
        alpha = torch.ones(1, h, w, 1, device=image.device, dtype=image.dtype)
        # 在边缘区域降低 alpha
        edge_mask_1ch = edge_mask.unsqueeze(-1)  # (1, H, W, 1)
        alpha = alpha * (1.0 - edge_mask_1ch * marker_alpha) + edge_mask_1ch * marker_alpha
        marked_image = torch.cat([marked_rgb, alpha], dim=-1)

        # --- 4. 应用 mask_alpha 到边缘遮罩输出 ---
        edge_mask_output = edge_mask * mask_alpha

        return (marked_image, edge_mask_output)


class PasteCroppedImageWithEdgeMarker:
    """将处理后的裁剪图像贴回原图，并在裁剪边缘生成标记遮罩和红色透明图层"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "边缘羽化像素数"
                }),
                "expand_outward": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "标记区域向外扩展（像素），覆盖原图侧"
                }),
                "expand_inward": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "标记区域向内扩展（像素），覆盖裁剪图侧"
                }),
                "marker_alpha": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "红色标记图层透明度"
                }),
                "mask_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "边缘遮罩透明度（用于生图模型修复）"
                }),
            },
            "optional": {
                "crop_position": ("STRING", {
                    "multiline": True,
                    "tooltip": "裁剪位置信息（JSON格式），与 mask 二选一"
                }),
                "mask": ("MASK", {
                    "tooltip": "边缘标记遮罩，沿其周边生成标记；与 crop_position 二选一"
                }),
            }
        }

    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("pasted_image", "edge_mask", "marked_image")

    def main(self, processed_image, original_image, feather,
             expand_outward, expand_inward, marker_alpha, mask_alpha,
             crop_position=None, mask=None):
        """
        贴回裁剪图像并生成边缘标记

        Args:
            processed_image: 处理后的裁剪图像
            original_image: 原图
            crop_position: 裁剪位置信息（JSON格式）
            feather: 边缘羽化像素数
            expand_outward: 标记区域向外扩展像素
            expand_inward: 标记区域向内扩展像素
            marker_alpha: 红色标记图层透明度
            mask_alpha: 边缘遮罩透明度

        Returns:
            pasted_image: 贴回后的完整图像
            edge_mask: 边缘区域遮罩
            marked_image: 带红色边缘标记的贴回图像（RGBA）
        """
        import json
        from PIL import Image, ImageFilter

        img_h, img_w = original_image.shape[1], original_image.shape[2]

        # --- 1. 解析位置信息（优先 crop_position，其次由 mask 计算）---
        if crop_position is not None and crop_position.strip():
            try:
                pos = json.loads(crop_position)
            except Exception:
                raise ValueError("crop_position 必须是有效的 JSON 字符串")

            if pos.get("is_empty", False):
                empty_mask = torch.zeros(1, img_h, img_w,
                                         device=original_image.device, dtype=torch.float32)
                return (original_image.clone(), empty_mask, original_image.clone())

            left = pos["left"]
            top = pos["top"]
            right = pos["right"]
            bottom = pos["bottom"]
        elif mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            mask_np = mask.cpu().numpy()
            rows = np.any(mask_np > 0, axis=1)
            cols = np.any(mask_np > 0, axis=0)
            if not np.any(rows) or not np.any(cols):
                empty_mask = torch.zeros(1, img_h, img_w,
                                         device=original_image.device, dtype=torch.float32)
                return (original_image.clone(), empty_mask, original_image.clone())
            top = int(np.argmax(rows))
            bottom = int(len(rows) - np.argmax(rows[::-1]) - 1)
            left = int(np.argmax(cols))
            right = int(len(cols) - np.argmax(cols[::-1]) - 1)
        else:
            raise ValueError("必须提供 crop_position 或 mask 其中之一")

        # --- 2. 缩放处理后图像到目标区域大小 ---
        target_width = right - left + 1
        target_height = bottom - top + 1
        proc_h, proc_w = processed_image.shape[1], processed_image.shape[2]

        if proc_h != target_height or proc_w != target_width:
            proc_np = processed_image[0].cpu().numpy()
            proc_np = (proc_np * 255).astype(np.uint8)
            proc_pil = Image.fromarray(proc_np)
            proc_pil = proc_pil.resize((target_width, target_height), Image.LANCZOS)
            proc_np = np.array(proc_pil).astype(np.float32) / 255.0
            processed_image = torch.from_numpy(proc_np)[None,].to(
                original_image.device, original_image.dtype)

        # --- 3. 贴回图像（与 PasteCroppedImage 相同逻辑）---
        result = original_image.clone()

        paste_mask = torch.ones(1, target_height, target_width,
                                device=original_image.device, dtype=torch.float32)
        if feather > 0:
            mask_np = paste_mask[0].cpu().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather))
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            paste_mask = torch.from_numpy(mask_np)[None,].to(
                original_image.device, torch.float32)

        paste_mask_3ch = paste_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        result[:, top:bottom+1, left:right+1, :] = (
            original_image[:, top:bottom+1, left:right+1, :] * (1 - paste_mask_3ch)
            + processed_image * paste_mask_3ch
        )

        # --- 4. 计算边缘区域遮罩 ---
        # 外边界（向外扩展后）
        outer_left = max(0, left - expand_outward)
        outer_top = max(0, top - expand_outward)
        outer_right = min(img_w - 1, right + expand_outward)
        outer_bottom = min(img_h - 1, bottom + expand_outward)

        # 内边界（向内收缩后）
        inner_left = min(img_w - 1, left + expand_inward)
        inner_top = min(img_h - 1, top + expand_inward)
        inner_right = max(-1, right - expand_inward)
        inner_bottom = max(-1, bottom - expand_inward)

        # 创建边缘遮罩 (1, H, W)
        edge_mask = torch.zeros(1, img_h, img_w,
                                device=original_image.device, dtype=torch.float32)
        edge_mask[:, outer_top:outer_bottom+1, outer_left:outer_right+1] = 1.0

        # 挖掉内部区域，只保留边缘环带
        if inner_left < inner_right and inner_top < inner_bottom:
            edge_mask[:, inner_top:inner_bottom+1, inner_left:inner_right+1] = 0.0

        # --- 5. 创建带红色标记的输出图像 ---
        result_rgb = result[:, :, :, :3].clone()  # 只取RGB
        _, h, w, _ = result_rgb.shape

        # 红色标记图层: (1, H, W, 3)，值为 (1, 0, 0)
        red_overlay = torch.zeros(1, h, w, 3, device=result.device, dtype=result.dtype)
        red_overlay[:, :, :, 0] = 1.0  # R通道

        # 扩展 edge_mask 到 3 通道
        edge_mask_3ch = edge_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        # 混合: result * (1 - edge_mask * marker_alpha) + red * edge_mask * marker_alpha
        marked_rgb = result_rgb * (1.0 - edge_mask_3ch * marker_alpha) \
                     + red_overlay * edge_mask_3ch * marker_alpha

        # 转为 RGBA，红色标记区域半透明
        alpha = torch.ones(1, h, w, 1, device=result.device, dtype=result.dtype)
        # 在边缘区域降低 alpha
        edge_mask_1ch = edge_mask.unsqueeze(-1)  # (1, H, W, 1)
        alpha = alpha * (1.0 - edge_mask_1ch * marker_alpha) + edge_mask_1ch * marker_alpha
        marked_image = torch.cat([marked_rgb, alpha], dim=-1)

        # --- 6. 应用 mask_alpha 到边缘遮罩输出 ---
        edge_mask_output = edge_mask * mask_alpha

        return (result, edge_mask_output, marked_image)

