from .mask_nodes import NODE_CLASS_MAPPINGS as MASK_NODE_CLASS_MAPPINGS
from .mask_nodes import NODE_DISPLAY_NAME_MAPPINGS as MASK_NODE_DISPLAY_NAME_MAPPINGS
from .image_nodes.image_ops import (
    CropImageByMask,
    CropImageWithWhiteBackground,
    ReplaceBackgroundWithWhite,
    ReplaceBackgroundWithWhiteExpand,
    VisualizeDetectionBox,
    FillMaskWithColor,
)

# 合并所有节点
NODE_CLASS_MAPPINGS = {
    **MASK_NODE_CLASS_MAPPINGS,
    "CropImageByMask": CropImageByMask,
    "CropImageWithWhiteBackground": CropImageWithWhiteBackground,
    "ReplaceBackgroundWithWhite": ReplaceBackgroundWithWhite,
    "ReplaceBackgroundWithWhiteExpand": ReplaceBackgroundWithWhiteExpand,
    "VisualizeDetectionBox": VisualizeDetectionBox,
    "FillMaskWithColor": FillMaskWithColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MASK_NODE_DISPLAY_NAME_MAPPINGS,
    "CropImageByMask": "按遮罩裁剪图片",
    "CropImageWithWhiteBackground": "裁剪图片并替换背景为白色",
    "ReplaceBackgroundWithWhite": "只替换背景为白色",
    "ReplaceBackgroundWithWhiteExpand": "替换背景为白色（可扩展空白）",
    "VisualizeDetectionBox": "可视化检测框",
    "FillMaskWithColor": "遮罩区域填充颜色",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
