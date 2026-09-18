import torch


class AlphaOverlayImage:
    """将带 Alpha 通道的副图像叠加到主图像上。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "main_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
            }
        }

    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    def main(self, main_image, overlay_image):
        overlay_alpha = overlay_image[..., 3:4]
        result = (
            main_image[..., :3] * (1.0 - overlay_alpha)
            + overlay_image[..., :3] * overlay_alpha
        )
        return (result.clamp(0.0, 1.0),)
