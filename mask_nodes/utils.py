import torch
import numpy as np
from PIL import Image, ImageFilter
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
    调整图片大小以适配目标尺寸（支持RGB和RGBA）

    Args:
        source_image: 源图片张量 shape (1, H, W, C) C=3或4
        target_width: 目标宽度
        target_height: 目标高度
        keep_aspect_ratio: 是否保持宽高比
        cover_mode: 覆盖模式 (True=完全覆盖可能裁剪, False=完全适应可能留空)

    Returns:
        调整大小后的图片张量
    """
    _, src_h, src_w, src_c = source_image.shape

    # 处理异常情况：如果输入图像尺寸为 1x1，直接返回目标尺寸的全1图像
    if src_h == 1 and src_w == 1:
        # 创建目标尺寸的全1图像
        result = torch.ones(1, target_height, target_width, src_c, device=source_image.device, dtype=torch.float32)
        return result

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

    # 分离RGB和Alpha通道（如果有）
    if src_c == 4:
        img_rgb = source_image[0, :, :, :3].cpu().numpy()
        img_alpha = source_image[0, :, :, 3:4].cpu().numpy()

        img_rgb = (img_rgb * 255).astype(np.uint8)
        img_alpha = (img_alpha * 255).astype(np.uint8)

        pil_rgb = Image.fromarray(img_rgb)
        pil_alpha = Image.fromarray(img_alpha.squeeze(-1))

        # 调整大小
        pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
        pil_alpha = pil_alpha.resize((new_w, new_h), Image.LANCZOS)

        # 转换回张量
        img_rgb_np = np.array(pil_rgb).astype(np.float32) / 255.0
        img_alpha_np = np.array(pil_alpha).astype(np.float32) / 255.0

        # 合并RGB和Alpha
        img_np = np.concatenate([img_rgb_np, img_alpha_np[..., None]], axis=-1)
    elif src_c == 1:
        # 单通道图像（如Alpha通道）
        img_np = source_image[0, :, :, 0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # 调整大小
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # 转换回张量，保持单通道
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        img_np = img_np[..., None]  # 添加通道维度
    else:
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
    将overlay图片合成到base图片的指定位置（基础版，支持RGBA输入）
    """
    left, top, right, bottom = target_bbox
    target_width = right - left + 1
    target_height = bottom - top + 1

    # 分离overlay的RGB和Alpha通道
    _, ov_h, ov_w, ov_c = overlay_image.shape

    # 处理异常情况：如果 overlay_image 的 H 或 W 为 1，可能是输入有问题
    if ov_h == 1 and ov_w == 1 and ov_c >= 3:
        # 这种情况通常是输入错误，创建一个默认的 RGB/RGBA 图像
        if ov_c == 4:
            overlay_rgb = torch.ones(1, 1, 1, 3, device=overlay_image.device, dtype=torch.float32)
            overlay_alpha = torch.ones(1, 1, 1, 1, device=overlay_image.device, dtype=torch.float32)
        else:
            overlay_rgb = overlay_image
            overlay_alpha = torch.ones(1, ov_h, ov_w, 1, device=overlay_image.device, dtype=torch.float32)
    elif ov_c == 4:
        overlay_rgb = overlay_image[:, :, :, :3]
        overlay_alpha = overlay_image[:, :, :, 3:4]
    else:
        overlay_rgb = overlay_image
        overlay_alpha = torch.ones(1, ov_h, ov_w, 1, device=overlay_image.device, dtype=torch.float32)

    resized_overlay_rgb = resize_image_to_fit(
        overlay_rgb,
        target_width,
        target_height,
        keep_aspect_ratio=True,
        cover_mode=cover_mode
    )
    resized_overlay_alpha = resize_image_to_fit(
        overlay_alpha,
        target_width,
        target_height,
        keep_aspect_ratio=True,
        cover_mode=cover_mode
    )

    _, overlay_h, overlay_w, _ = resized_overlay_rgb.shape

    # 处理遮罩 shape 和数据类型
    if overlay_mask.dim() == 2:
        overlay_mask = overlay_mask.unsqueeze(0)
    elif overlay_mask.dim() == 3 and overlay_mask.shape[0] == 1:
        pass  # 已经是 (1, H, W)
    elif overlay_mask.dim() == 3 and overlay_mask.shape[1] == 1 and overlay_mask.shape[2] == 1:
        # shape 是 (1, 1, 1) 这种特殊情况，扩展为全1遮罩
        overlay_mask = torch.ones(1, overlay_h, overlay_w, device=overlay_mask.device, dtype=torch.float32)

    # 确保遮罩是 float32 类型
    if overlay_mask.dtype != torch.float32:
        overlay_mask = overlay_mask.float()

    mask_np = overlay_mask[0].cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)
    mask_pil = mask_pil.resize((overlay_w, overlay_h), Image.LANCZOS)

    if feather > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather))

    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    resized_mask = torch.from_numpy(mask_np)[None,]

    # 合并遮罩：输入遮罩 × overlay的alpha通道
    combined_mask = resized_mask * resized_overlay_alpha.squeeze(-1)

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

        resized_overlay_rgb = resized_overlay_rgb[:, crop_top:crop_bottom, crop_left:crop_right, :]
        combined_mask = combined_mask[:, crop_top:crop_bottom, crop_left:crop_right]

        _, overlay_h, overlay_w, _ = resized_overlay_rgb.shape

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

    mask_3ch = combined_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

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
        resized_overlay_rgb[:, :paste_h, :paste_w, :] * mask_3ch[:, :paste_h, :paste_w, :]
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
    scale_factor: float = 0.0,
    skip_initial_resize: bool = False
) -> torch.Tensor:
    """
    将overlay图片合成到base图片的指定位置（增强版，含偏移/缩放/裁切，支持RGBA输入）
    """
    left, top, right, bottom = target_bbox
    target_width = right - left + 1
    target_height = bottom - top + 1

    # 分离overlay的RGB和Alpha通道
    _, ov_h, ov_w, ov_c = overlay_image.shape

    # 处理异常情况：如果 overlay_image 的 H 或 W 为 1，可能是输入有问题
    if ov_h == 1 and ov_w == 1 and ov_c >= 3:
        # 这种情况通常是输入错误，创建一个默认的 RGB/RGBA 图像
        if ov_c == 4:
            overlay_rgb = torch.ones(1, 1, 1, 3, device=overlay_image.device, dtype=torch.float32)
            overlay_alpha = torch.ones(1, 1, 1, 1, device=overlay_image.device, dtype=torch.float32)
        else:
            overlay_rgb = overlay_image
            overlay_alpha = torch.ones(1, ov_h, ov_w, 1, device=overlay_image.device, dtype=torch.float32)
    elif ov_c == 4:
        overlay_rgb = overlay_image[:, :, :, :3]
        overlay_alpha = overlay_image[:, :, :, 3:4]
    else:
        overlay_rgb = overlay_image
        overlay_alpha = torch.ones(1, ov_h, ov_w, 1, device=overlay_image.device, dtype=torch.float32)

    if not skip_initial_resize:
        resized_overlay_rgb = resize_image_to_fit(
            overlay_rgb,
            target_width,
            target_height,
            keep_aspect_ratio=True,
            cover_mode=cover_mode
        )
        resized_overlay_alpha = resize_image_to_fit(
            overlay_alpha,
            target_width,
            target_height,
            keep_aspect_ratio=True,
            cover_mode=cover_mode
        )
    else:
        resized_overlay_rgb = overlay_rgb
        resized_overlay_alpha = overlay_alpha

    if scale_factor != 0.0:
        _, overlay_h, overlay_w, _ = resized_overlay_rgb.shape
        scale_multiplier = 1.0 + (scale_factor / 100.0)
        new_w = max(1, int(overlay_w * scale_multiplier))
        new_h = max(1, int(overlay_h * scale_multiplier))

        # 分别缩放RGB和Alpha
        img_rgb_np = resized_overlay_rgb[0].cpu().numpy()
        img_rgb_np = (img_rgb_np * 255).astype(np.uint8)
        pil_rgb = Image.fromarray(img_rgb_np)
        pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
        img_rgb_np = np.array(pil_rgb).astype(np.float32) / 255.0
        resized_overlay_rgb = torch.from_numpy(img_rgb_np)[None,]

        img_alpha_np = resized_overlay_alpha[0].cpu().numpy()
        img_alpha_np = (img_alpha_np * 255).astype(np.uint8)
        pil_alpha = Image.fromarray(img_alpha_np.squeeze(-1))
        pil_alpha = pil_alpha.resize((new_w, new_h), Image.LANCZOS)
        img_alpha_np = np.array(pil_alpha).astype(np.float32) / 255.0
        resized_overlay_alpha = torch.from_numpy(img_alpha_np)[None, ..., None]

    _, overlay_h, overlay_w, _ = resized_overlay_rgb.shape

    # 处理遮罩 shape 和数据类型
    if overlay_mask.dim() == 2:
        overlay_mask = overlay_mask.unsqueeze(0)
    elif overlay_mask.dim() == 3 and overlay_mask.shape[0] == 1:
        pass  # 已经是 (1, H, W)
    elif overlay_mask.dim() == 3 and overlay_mask.shape[1] == 1 and overlay_mask.shape[2] == 1:
        # shape 是 (1, 1, 1) 这种特殊情况，扩展为全1遮罩
        overlay_mask = torch.ones(1, overlay_h, overlay_w, device=overlay_mask.device, dtype=torch.float32)

    # 确保遮罩是 float32 类型
    if overlay_mask.dtype != torch.float32:
        overlay_mask = overlay_mask.float()

    mask_np = overlay_mask[0].cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)
    mask_pil = mask_pil.resize((overlay_w, overlay_h), Image.LANCZOS)

    if feather > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather))

    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    resized_mask = torch.from_numpy(mask_np)[None,]

    # 合并遮罩：输入遮罩 × overlay的alpha通道
    combined_mask = resized_mask * resized_overlay_alpha.squeeze(-1)

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

        resized_overlay_rgb = resized_overlay_rgb[:, crop_top:crop_bottom, crop_left:crop_right, :]
        combined_mask = combined_mask[:, crop_top:crop_bottom, crop_left:crop_right]

        _, overlay_h, overlay_w, _ = resized_overlay_rgb.shape

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

    mask_3ch = combined_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

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
        resized_overlay_rgb[:, src_start_y:src_start_y+paste_h, src_start_x:src_start_x+paste_w, :] * mask_3ch[:, src_start_y:src_start_y+paste_h, src_start_x:src_start_x+paste_w, :]
    )

    return result
