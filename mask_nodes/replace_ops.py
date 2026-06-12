import torch
from .utils import get_mask_bounding_box, composite_images, composite_images_v2


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


class ImageReplaceWithMaskV3:
    """
    根据遮罩智能替换图片物体（V3版，含扩展空白/自适应扩展）
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
                "auto_expand_height": ("BOOLEAN", {
                    "default": False,
                    "label_on": "高度自适应扩展",
                    "label_off": "关闭高度自适应"
                }),
                "auto_expand_width": ("BOOLEAN", {
                    "default": False,
                    "label_on": "宽度自适应扩展",
                    "label_off": "关闭宽度自适应"
                }),
                "enable_shrink_after_fit": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用缩小",
                    "label_off": "不缩小"
                }),
                "shrink_ratio": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
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
        auto_expand_height,
        auto_expand_width,
        enable_shrink_after_fit,
        shrink_ratio,
        replace_mask=None
    ):
        """
        替换图片中的物体，支持扩展空白和自适应扩展
        
        Args:
            base_image: 基础图片
            base_mask: 基础图片的遮罩（标识要替换的区域）
            replace_image: 替换源图片
            keep_aspect_ratio: 是否保持宽高比
            cover_mode: 覆盖模式 (True=完全覆盖可能裁剪, False=完全适应可能留空)
            alignment: 对齐方式 (center/top/bottom/left/right)
            feather: 边缘羽化程度
            offset_left: 向左偏移像素数
            offset_right: 向右偏移像素数
            offset_up: 向上偏移像素数
            offset_down: 向下偏移像素数
            allow_crop: 是否允许裁切 (True=允许裁切, False=禁用裁切保持完整)
            auto_expand_height: 是否启用高度自适应扩展
            auto_expand_width: 是否启用宽度自适应扩展（高度和宽度都开时优先高度）
            enable_shrink_after_fit: 是否在贴合后再额外按比例整体缩小
            shrink_ratio: 贴合后缩小的倍数（0.01-1.0）
            replace_mask: 可选的替换图片遮罩，如果不提供则使用整个图片
            
        Returns:
            合成后的图片
        """
        import numpy as np
        from PIL import Image
        from .utils import resize_image_to_fit
        
        # 获取基础图片遮罩的边界框
        target_bbox = get_mask_bounding_box(base_mask)
        left, top, right, bottom = target_bbox
        target_width = right - left + 1
        target_height = bottom - top + 1
        
        # 计算替换物体的实际尺寸（基于 replace_mask，若无则整图）
        replace_h_obj, replace_w_obj = None, None
        if replace_mask is None:
            _, _rh, _rw, _ = replace_image.shape
            replace_h_obj, replace_w_obj = _rh, _rw
        else:
            if replace_mask.dim() == 2:
                replace_mask = replace_mask.unsqueeze(0)
            rb_left, rb_top, rb_right, rb_bottom = get_mask_bounding_box(replace_mask)
            if rb_right > rb_left and rb_bottom > rb_top:
                replace_w_obj = rb_right - rb_left + 1
                replace_h_obj = rb_bottom - rb_top + 1
            else:
                _, _rh, _rw, _ = replace_image.shape
                replace_h_obj, replace_w_obj = _rh, _rw
        
        # 自适应扩展：先处理"高度+宽度都开"的完全贴合模式，然后是单独高度或单独宽度模式
        def pad_image_and_mask(img, msk, pad_l, pad_r, pad_u, pad_d):
            if msk is None:
                _, h, w, _ = img.shape
                msk = torch.ones(1, h, w, device=img.device, dtype=img.dtype)
            if msk.dim() == 2:
                msk = msk.unsqueeze(0)
            _, h, w, c = img.shape
            new_w = w + pad_l + pad_r
            new_h = h + pad_u + pad_d
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            paste_x = max(0, pad_l)
            paste_y = max(0, pad_u)
            src_start_x = max(0, -pad_l)
            src_start_y = max(0, -pad_u)
            src_end_x = w - max(0, -pad_r)
            src_end_y = h - max(0, -pad_d)
            paste_width = src_end_x - src_start_x
            paste_height = src_end_y - src_start_y

            # 支持RGB和RGBA
            if c == 4:
                # RGBA: 白色背景RGB + 不透明Alpha（补白区域需要显示白色）
                white_bg = torch.ones(1, new_h, new_w, 3, device=img.device, dtype=img.dtype)
                opaque_alpha = torch.ones(1, new_h, new_w, 1, device=img.device, dtype=img.dtype)  # 补白区域Alpha=1（不透明）
                if paste_width > 0 and paste_height > 0:
                    white_bg[:, paste_y:paste_y+paste_height, paste_x:paste_x+paste_width, :] = img[:, src_start_y:src_end_y, src_start_x:src_end_x, :3]
                    opaque_alpha[:, paste_y:paste_y+paste_height, paste_x:paste_x+paste_width, :] = img[:, src_start_y:src_end_y, src_start_x:src_end_x, 3:4]
                # 合并RGB和Alpha
                white_bg = torch.cat([white_bg, opaque_alpha], dim=-1)
            else:
                white_bg = torch.ones(1, new_h, new_w, 3, device=img.device, dtype=img.dtype)
                if paste_width > 0 and paste_height > 0:
                    white_bg[:, paste_y:paste_y+paste_height, paste_x:paste_x+paste_width, :] = img[:, src_start_y:src_end_y, src_start_x:src_end_x, :]

            new_mask = torch.zeros(1, new_h, new_w, device=msk.device, dtype=msk.dtype)
            if paste_width > 0 and paste_height > 0:
                new_mask[:, paste_y:paste_y+paste_height, paste_x:paste_x+paste_width] = msk[:, src_start_y:src_end_y, src_start_x:src_end_x]
            # 为了让补白区域在合成时显示为白色，需要让补白区域对应的mask为1
            # 计算补白区域：新画布减去粘贴区域的四个边带
            # 顶部补白
            if paste_y > 0:
                new_mask[:, 0:paste_y, :] = 1
            # 底部补白
            bottom_start = paste_y + max(0, paste_height)
            if bottom_start < new_h:
                new_mask[:, bottom_start:new_h, :] = 1
            # 左侧补白
            if paste_x > 0:
                new_mask[:, :, 0:paste_x] = 1
            # 右侧补白
            right_start = paste_x + max(0, paste_width)
            if right_start < new_w:
                new_mask[:, :, right_start:new_w] = 1
            return white_bg, new_mask

        # 公共缩放函数（用于高度/宽度联合模式，支持RGBA）
        def resize_tensor_img(tensor_img, new_w, new_h):
            _, h, w, c = tensor_img.shape
            if c == 4:
                # RGBA: 分别缩放RGB和Alpha
                img_rgb = (tensor_img[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
                img_alpha = (tensor_img[0, :, :, 3:4].cpu().numpy() * 255).astype(np.uint8)

                pil_rgb = Image.fromarray(img_rgb)
                pil_alpha = Image.fromarray(img_alpha.squeeze(-1))

                pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
                pil_alpha = pil_alpha.resize((new_w, new_h), Image.LANCZOS)

                np_rgb = np.array(pil_rgb).astype(np.float32) / 255.0
                np_alpha = np.array(pil_alpha).astype(np.float32) / 255.0

                np_img = np.concatenate([np_rgb, np_alpha[..., None]], axis=-1)
            else:
                np_img = (tensor_img[0].cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(np_img)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                np_img = np.array(pil_img).astype(np.float32) / 255.0

            return torch.from_numpy(np_img)[None,].to(tensor_img.device, tensor_img.dtype)
        
        def resize_tensor_mask(msk, new_w, new_h):
            np_m = (msk[0].cpu().numpy() * 255).astype(np.uint8)
            pil_m = Image.fromarray(np_m)
            pil_m = pil_m.resize((new_w, new_h), Image.LANCZOS)
            np_m = np.array(pil_m).astype(np.float32) / 255.0
            return torch.from_numpy(np_m)[None,].to(msk.device, msk.dtype)

        # 情况1：高度 + 宽度自适应都开启 -> 只允许缩放 + 白边，不裁切，最终画布严格贴合 base_mask 的 bbox
        if auto_expand_height and auto_expand_width and replace_h_obj > 0 and replace_w_obj > 0:
            # 以物体尺寸为基准，计算统一缩放比例；同时保证"整图缩放后不会超过 target 尺寸"
            _, rh, rw, _ = replace_image.shape
            scale_obj = min(target_width / replace_w_obj, target_height / replace_h_obj)
            scale_img = min(target_width / rw, target_height / rh)
            scale = max(min(scale_obj, scale_img), 1e-6)
            eff_ratio = shrink_ratio if enable_shrink_after_fit else 1.0
            eff_ratio = max(0.01, min(1.0, float(eff_ratio)))
            scale_final = scale * eff_ratio
            new_obj_w = max(1, int(round(replace_w_obj * scale_final)))
            new_obj_h = max(1, int(round(replace_h_obj * scale_final)))
            
            # 按统一比例缩放整张 replace_image / replace_mask（避免裁切物体且不超框）
            new_w_img = max(1, int(round(rw * scale_final)))
            new_h_img = max(1, int(round(rh * scale_final)))
            replace_image = resize_tensor_img(replace_image, new_w_img, new_h_img)
            if replace_mask is None:
                replace_mask = torch.ones(1, new_h_img, new_w_img, device=replace_image.device, dtype=replace_image.dtype)
            else:
                if replace_mask.dim() == 2:
                    replace_mask = replace_mask.unsqueeze(0)
                replace_mask = resize_tensor_mask(replace_mask, new_w_img, new_h_img)
            
            # 在目标画布(target_width, target_height)中居中放置缩放后的图像（只加白边，不再裁切）
            pad_l = max(0, (target_width - new_w_img) // 2)
            pad_r = max(0, target_width - new_w_img - pad_l)
            pad_u = max(0, (target_height - new_h_img) // 2)
            pad_d = max(0, target_height - new_h_img - pad_u)
            replace_image, replace_mask = pad_image_and_mask(
                replace_image, replace_mask, pad_l, pad_r, pad_u, pad_d
            )

        # 情况2：只开启高度自适应 -> 高度匹配 base_mask，高度缩放后用左右白边补足覆盖宽度
        elif auto_expand_height:
            if replace_h_obj > 0:
                scale_h = target_height / replace_h_obj
                if scale_h > 0:
                    # 应用缩小比例
                    eff_ratio = shrink_ratio if enable_shrink_after_fit else 1.0
                    eff_ratio = max(0.01, min(1.0, float(eff_ratio)))
                    scale_h_final = scale_h * eff_ratio
                    
                    _, rh, rw, _ = replace_image.shape
                    new_h = max(1, int(round(target_height * eff_ratio)))
                    new_w = max(1, int(round(rw * scale_h_final)))
                    replace_image = resize_tensor_img(replace_image, new_w, new_h)
                    if replace_mask is None:
                        replace_mask = torch.ones(1, new_h, new_w, device=replace_image.device, dtype=replace_image.dtype)
                    else:
                        if replace_mask.dim() == 2:
                            replace_mask = replace_mask.unsqueeze(0)
                        replace_mask = resize_tensor_mask(replace_mask, new_w, new_h)
                    
                    # 宽度自适应：按"物体"宽度而不是整图宽度来判断是否需要左右补白
                    # 重新计算缩放后物体在 replace_mask 中的 bbox
                    rb_l2, rb_t2, rb_r2, rb_b2 = get_mask_bounding_box(replace_mask)
                    obj_width_after = max(0, rb_r2 - rb_l2 + 1)
                    pad_l = pad_r = 0
                    if new_w < target_width:
                        pad_total = target_width - new_w
                        pad_l = pad_total // 2
                        pad_r = pad_total - pad_l
                    replace_image, replace_mask = pad_image_and_mask(
                        replace_image, replace_mask, pad_l, pad_r, 0, 0
                    )
        
        # 情况3：只开启宽度自适应 -> 宽度为基准，按比例补上下白边，使整体比例接近 base_mask
        if auto_expand_width and not auto_expand_height:
            # 应用缩小比例
            eff_ratio = shrink_ratio if enable_shrink_after_fit else 1.0
            eff_ratio = max(0.01, min(1.0, float(eff_ratio)))
            
            _, rh, rw, _ = replace_image.shape
            # 先按缩小比例缩放替换图
            if eff_ratio < 1.0:
                new_rw = max(1, int(round(rw * eff_ratio)))
                new_rh = max(1, int(round(rh * eff_ratio)))
                replace_image = resize_tensor_img(replace_image, new_rw, new_rh)
                if replace_mask is not None:
                    if replace_mask.dim() == 2:
                        replace_mask = replace_mask.unsqueeze(0)
                    replace_mask = resize_tensor_mask(replace_mask, new_rw, new_rh)
                _, rh, rw, _ = replace_image.shape
            
            target_aspect = target_width / target_height if target_height > 0 else 1.0
            needed_h = int(round(rw / target_aspect)) if target_width > 0 else rh
            extra_h = max(0, needed_h - rh)
            pad_u = extra_h // 2
            pad_d = extra_h - pad_u
            replace_image, replace_mask = pad_image_and_mask(replace_image, replace_mask, 0, 0, pad_u, pad_d)

        # 已移除手动 expand_* 相关逻辑
        
        # 处理replace_mask
        if replace_mask is None:
            _, h, w, _ = replace_image.shape
            replace_mask = torch.ones(1, h, w)
        
        if replace_mask.dim() == 2:
            replace_mask = replace_mask.unsqueeze(0)
        
        cropped_replace = replace_image
        cropped_mask = replace_mask
        
        # 计算偏移量
        offset_x = offset_right - offset_left
        offset_y = offset_down - offset_up
        
        auto_on = auto_expand_height or auto_expand_width
        final_allow_crop = False if auto_on else allow_crop
        
        # 合成图片
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
            allow_crop=final_allow_crop,
            skip_initial_resize=auto_on
        )
        
        return (result,)
