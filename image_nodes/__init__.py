from .image_ops import (
    CropImageByMask,
    CropImageWithWhiteBackground,
    CropImageWithWhiteBackgroundV2,
    ReplaceBackgroundWithWhite,
    ReplaceBackgroundWithWhiteExpand,
    VisualizeDetectionBox,
    FillMaskWithColor,
    CropImageWithPosition,
    PasteCroppedImage,
    PasteCroppedImageWithEdgeMarker,
    MaskToCropPosition,
    MaskEdgeMarker,
)

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "CropImageByMask": CropImageByMask,
    "CropImageWithWhiteBackground": CropImageWithWhiteBackground,
    "CropImageWithWhiteBackgroundV2": CropImageWithWhiteBackgroundV2,
    "ReplaceBackgroundWithWhite": ReplaceBackgroundWithWhite,
    "ReplaceBackgroundWithWhiteExpand": ReplaceBackgroundWithWhiteExpand,
    "VisualizeDetectionBox": VisualizeDetectionBox,
    "FillMaskWithColor": FillMaskWithColor,
    "CropImageWithPosition": CropImageWithPosition,
    "PasteCroppedImage": PasteCroppedImage,
    "PasteCroppedImageWithEdgeMarker": PasteCroppedImageWithEdgeMarker,
    "MaskToCropPosition": MaskToCropPosition,
    "MaskEdgeMarker": MaskEdgeMarker,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "CropImageByMask": "按遮罩裁剪图片",
    "CropImageWithWhiteBackground": "裁剪图片并替换背景为白色",
    "CropImageWithWhiteBackgroundV2": "裁剪图片并替换背景为白色（V2，可强制1:1）",
    "ReplaceBackgroundWithWhite": "只替换背景为白色",
    "ReplaceBackgroundWithWhiteExpand": "替换背景为白色（可扩展空白）",
    "VisualizeDetectionBox": "可视化检测框",
    "FillMaskWithColor": "遮罩区域填充颜色",
    "CropImageWithPosition": "裁剪图像（带位置信息）",
    "PasteCroppedImage": "贴回裁剪图像",
    "PasteCroppedImageWithEdgeMarker": "贴回裁剪图像（边缘标记）",
    "MaskToCropPosition": "遮罩转裁剪位置",
    "MaskEdgeMarker": "遮罩边缘标记",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
