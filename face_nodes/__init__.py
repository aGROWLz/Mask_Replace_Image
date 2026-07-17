from .face_ops import (
    FaceReplaceWithLandmark,
)

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "FaceReplaceWithLandmark": FaceReplaceWithLandmark,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceReplaceWithLandmark": "人脸关键点替换",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
