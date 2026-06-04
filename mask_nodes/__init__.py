from .utils import (
    get_mask_bounding_box,
    crop_image_by_mask,
    resize_image_to_fit,
    composite_images,
    composite_images_v2,
)

from .mask_ops import (
    MaskBoundingBox,
    MergeMasks,
    MergeMasksV2,
    MergeMasksDelete,
    SelectLargestMask,
    SelectLargestMaskByArea,
)

from .replace_ops import (
    ImageReplaceWithMask,
    ImageReplaceWithMaskV2,
    ImageReplaceWithMaskV3,
)

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskBoundingBox": MaskBoundingBox,
    "MergeMasks": MergeMasks,
    "MergeMasksV2": MergeMasksV2,
    "MergeMasksDelete": MergeMasksDelete,
    "SelectLargestMask": SelectLargestMask,
    "SelectLargestMaskByArea": SelectLargestMaskByArea,
    "ImageReplaceWithMask": ImageReplaceWithMask,
    "ImageReplaceWithMaskV2": ImageReplaceWithMaskV2,
    "ImageReplaceWithMaskV3": ImageReplaceWithMaskV3,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBoundingBox": "提取遮罩边界框",
    "MergeMasks": "合并遮罩",
    "MergeMasksV2": "合并遮罩 V2",
    "MergeMasksDelete": "合并遮罩（删除）",
    "SelectLargestMask": "筛选最大遮罩",
    "SelectLargestMaskByArea": "筛选最大遮罩（按面积）",
    "ImageReplaceWithMask": "智能物体替换",
    "ImageReplaceWithMaskV2": "智能物体替换 V2",
    "ImageReplaceWithMaskV3": "智能物体替换 V3",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
