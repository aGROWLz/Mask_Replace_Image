"""图像颜色匹配节点（复刻自 comfyui-easy-use 的 Image Color Match 节点）。

支持两种颜色匹配路径：
1. wavelet / adain：基于 torch 的轻量实现，无需第三方依赖；
2. mkl / hm / reinhard / mvgd / hm-mvgd-hm / hm-mkl-hm：依赖 color-matcher 库。
"""
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F


# ---------- 基础转换 ----------
def tensor2pil(image):
    """ComfyUI IMAGE 张量 (1, H, W, C) -> PIL Image"""
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    """PIL Image -> ComfyUI IMAGE 张量 (1, H, W, C)"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def _pil_to_chw_tensor(pil_img):
    """PIL Image -> (1, C, H, W) float32 张量"""
    arr = np.array(pil_img).astype(np.float32) / 255.0  # (H, W, C)
    arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)


def _chw_tensor_to_pil(tensor):
    """(1, C, H, W) float32 张量 -> PIL Image"""
    arr = tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


# ---------- AdaIN ----------
def _calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, "输入特征应为 4D 张量"
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def _adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adain_color_fix(target, source):
    """将 target 的颜色/光照对齐到 source（AdaIN 方法）"""
    target_tensor = _pil_to_chw_tensor(target)
    source_tensor = _pil_to_chw_tensor(source)
    result_tensor = _adaptive_instance_normalization(target_tensor, source_tensor)
    return _chw_tensor_to_pil(result_tensor)


# ---------- Wavelet ----------
def _wavelet_blur(image, radius):
    # input shape: (1, C, H, W)
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None]
    kernel = kernel.repeat(image.size(1), 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    output = F.conv2d(image, kernel, groups=image.size(1), dilation=radius)
    return output


def _wavelet_decomposition(image, levels=5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = _wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def _wavelet_reconstruction(content_feat, style_feat):
    content_high_freq, content_low_freq = _wavelet_decomposition(content_feat)
    del content_low_freq
    style_high_freq, style_low_freq = _wavelet_decomposition(style_feat)
    del style_high_freq
    return content_high_freq + style_low_freq


def wavelet_color_fix(target, source):
    """将 target 的颜色对齐到 source（Wavelet 方法）"""
    source = source.resize(target.size, resample=Image.Resampling.LANCZOS)
    target_tensor = _pil_to_chw_tensor(target)
    source_tensor = _pil_to_chw_tensor(source)
    result_tensor = _wavelet_reconstruction(target_tensor, source_tensor)
    return _chw_tensor_to_pil(result_tensor)


# ---------- 节点 ----------
class ImageColorMatch:
    """图像颜色匹配：将参考图的颜色风格迁移到目标图"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (["wavelet", "adain", "mkl", "hm", "reinhard", "mvgd",
                            "hm-mvgd-hm", "hm-mkl-hm"],),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    def main(self, image_ref, image_target, method, mask=None):
        """颜色匹配主流程

        Args:
            image_ref: 参考图（颜色来源）
            image_target: 目标图（被调整颜色的图）
            method: 颜色匹配算法
            mask: 可选遮罩，只对遮罩区域应用颜色匹配；不接入则处理整张图

        Returns:
            颜色匹配后的图片
        """
        if method in ["wavelet", "adain"]:
            if method == "wavelet":
                result_images = wavelet_color_fix(tensor2pil(image_target), tensor2pil(image_ref))
            else:
                result_images = adain_color_fix(tensor2pil(image_target), tensor2pil(image_ref))
            new_images = pil2tensor(result_images)
        else:
            try:
                from color_matcher import ColorMatcher
            except ImportError:
                raise ImportError(
                    "使用该方法需要安装 color-matcher 库，请运行：pip install color-matcher"
                )

            image_ref = image_ref.cpu()
            image_target = image_target.cpu()
            batch_size = image_target.size(0)
            out = []
            images_target = image_target.squeeze()
            images_ref = image_ref.squeeze()

            image_ref_np = images_ref.numpy()
            images_target_np = images_target.numpy()
            if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
                raise ValueError(
                    "ColorMatch: 参考图只能是单张，或与目标图 batch 数量一致"
                )

            cm = ColorMatcher()
            for i in range(batch_size):
                image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
                image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
                try:
                    image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
                except BaseException as e:
                    print(f"Error occurred during transfer: {e}")
                    break
                out.append(torch.from_numpy(image_result))

            new_images = torch.stack(out, dim=0).to(torch.float32)

        # 若接入遮罩，则仅对遮罩区域应用颜色匹配，其余区域保留原图
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # (1, H, W)
            _, h, w, c = new_images.shape
            if mask.shape[1] != h or mask.shape[2] != w:
                mask = F.interpolate(
                    mask.unsqueeze(1), size=(h, w), mode="bilinear"
                ).squeeze(1)
            if mask.shape[0] == 1 and new_images.shape[0] > 1:
                mask = mask.repeat(new_images.shape[0], 1, 1)
            mask = mask.to(device=new_images.device, dtype=new_images.dtype)
            mask_c = mask.unsqueeze(-1).repeat(1, 1, 1, c)  # (B, H, W, C)
            image_target = image_target.to(device=new_images.device, dtype=new_images.dtype)
            new_images = new_images * mask_c + image_target * (1.0 - mask_c)

        return (new_images,)
