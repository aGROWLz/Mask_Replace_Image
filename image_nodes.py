import torch


def _parse_switch(s: str) -> bool:
    """
    从字符串解析开关：
    - 包含 "NO" (不区分大小写) 则关闭
    - 否则若包含 "YES" 则开启
    - 其他情况默认关闭
    """
    if s is None:
        return False
    text = str(s).upper()
    if "NO" in text:
        return False
    if "YES" in text:
        return True
    return False


class MirrorImageWithSwitch:
    """根据字符串开关选择是否对图片做水平镜像"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "switch": ("STRING", {
                    "default": "YES",
                    "multiline": False,
                    "placeholder": "YES 开启镜像 / NO 关闭镜像"
                }),
            }
        }
    
    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    def main(self, image, switch):
        enabled = _parse_switch(switch)
        if not enabled:
            return (image.clone(),)
        
        # 水平镜像：宽度维度为 dim=2
        mirrored = torch.flip(image, dims=[2])
        return (mirrored,)


NODE_CLASS_MAPPINGS = {
    "MirrorImageWithSwitch": MirrorImageWithSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MirrorImageWithSwitch": "镜像图片（开关）",
}

