import torch
import numpy as np
import cv2


class FaceReplaceWithLandmark:
    """基于 insightface 106 点人脸关键点的智能换脸节点

    使用 insightface 检测两张图片的人脸关键点，
    根据关键点自动计算缩放/旋转/平移，将源人脸对齐后覆盖到目标人脸上。
    """

    _face_app = None

    @classmethod
    def _get_face_app(cls):
        """延迟加载 insightface，避免每次调用都重新初始化"""
        if cls._face_app is None:
            import os
            import insightface
            from insightface.app import FaceAnalysis

            # 模型下载到插件目录 face_nodes/models 下
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(models_dir, exist_ok=True)

            cls._face_app = FaceAnalysis(name='buffalo_l', root=models_dir)
            cls._face_app.prepare(ctx_id=0, det_size=(640, 640))

            # 检测并打印实际使用的推理后端
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                print("[FaceReplace] insightface 使用 GPU 推理 (CUDA)")
            elif 'ROCMExecutionProvider' in providers:
                print("[FaceReplace] insightface 使用 GPU 推理 (ROCm)")
            else:
                print("[FaceReplace] insightface 使用 CPU 推理")
        return cls._face_app

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "feather": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "边缘羽化像素数，让覆盖过渡更自然"
                }),
                "scale_adjust": ("FLOAT", {
                    "default": 0.0,
                    "min": -50.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "手动微调缩放比例(%)，正数放大负数缩小"
                }),
                "mask_expand": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "遮罩向外扩展比例（相对人脸尺寸）"
                }),
                "enable_rotation": ("BOOLEAN", {
                    "default": True,
                    "label_on": "角度对齐",
                    "label_off": "仅缩放平移",
                    "tooltip": "开启后根据人脸关键点自动旋转对齐角度"
                }),
            },
        }

    CATEGORY = "image"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",)

    def main(self, target_image, source_image, feather, scale_adjust, mask_expand, enable_rotation):
        # ---- 步骤1: ComfyUI tensor → numpy uint8 BGR ----
        target_np = (target_image[0].cpu().numpy() * 255).astype(np.uint8)
        source_np = (source_image[0].cpu().numpy() * 255).astype(np.uint8)

        target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
        source_bgr = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)

        # ---- 步骤2: 人脸检测 ----
        app = self._get_face_app()
        target_faces = app.get(target_bgr)
        source_faces = app.get(source_bgr)

        if len(target_faces) == 0 or len(source_faces) == 0:
            return (target_image.clone(),)

        # 取面积最大的人脸
        def face_area(f):
            return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])

        target_face = max(target_faces, key=face_area)
        source_face = max(source_faces, key=face_area)

        # ---- 步骤3: 提取 5 个关键点 ----
        src_kps = source_face.kps[:, :2].astype(np.float32)
        dst_kps = target_face.kps[:, :2].astype(np.float32)

        # ---- 步骤4: 根据开关计算变换矩阵 ----
        if enable_rotation:
            M, _ = cv2.estimateAffinePartial2D(src_kps, dst_kps, method=cv2.RANSAC)
        else:
            M = self._estimate_scale_translate_only(src_kps, dst_kps)

        if M is None:
            return (target_image.clone(),)

        # ---- 步骤5: 手动微调缩放 ----
        if scale_adjust != 0.0:
            k = 1.0 + scale_adjust / 100.0
            old_M22 = M[:, :2].copy()
            M[:, :2] *= k
            src_center = np.mean(src_kps, axis=0)
            M[:, 2] += (1.0 - k) * (old_M22 @ src_center)

        # ---- 步骤6: 生成源人脸遮罩（椭圆覆盖全脸含额头） ----
        src_mask = self._create_source_mask(source_face, source_bgr.shape[:2], mask_expand)

        # ---- 步骤7: 用同一变换矩阵对齐源图和遮罩 ----
        th, tw = target_bgr.shape[:2]
        warped = cv2.warpAffine(
            source_bgr, M, (tw, th),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv2.warpAffine(
            src_mask, M, (tw, th),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # ---- 步骤8: 羽化 + 融合 ----
        if feather > 0:
            ksize = feather * 2 + 1
            warped_mask = cv2.GaussianBlur(warped_mask, (ksize, ksize), feather)

        mask_3ch = warped_mask[:, :, np.newaxis]
        result = (warped.astype(np.float32) * mask_3ch +
                  target_bgr.astype(np.float32) * (1.0 - mask_3ch))
        result = np.clip(result, 0, 255).astype(np.uint8)

        # ---- 步骤9: 转回 ComfyUI 格式 ----
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_tensor = torch.from_numpy(
            result_rgb.astype(np.float32) / 255.0
        )[None,]

        return (result_tensor,)

    # ==================== 工具方法 ====================

    def _create_source_mask(self, face, img_shape, expand_ratio):
        """根据源人脸的 bbox 生成椭圆遮罩，天然覆盖额头"""
        h, w = img_shape
        bbox = face.bbox.astype(np.int32)
        x1, y1, x2, y2 = bbox[:4]

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        # 扩张遮罩
        face_h = y2 - y1
        expand_px = max(3, int(face_h * expand_ratio))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand_px, expand_px)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    @staticmethod
    def _estimate_scale_translate_only(src_kps, dst_kps):
        """仅计算缩放+平移（无旋转），保留源人脸原始角度"""
        src_center = np.mean(src_kps, axis=0)
        dst_center = np.mean(dst_kps, axis=0)

        src_dist = np.mean([np.linalg.norm(p - src_center) for p in src_kps])
        dst_dist = np.mean([np.linalg.norm(p - dst_center) for p in dst_kps])
        scale = dst_dist / src_dist if src_dist > 0 else 1.0

        M = np.zeros((2, 3), dtype=np.float32)
        M[0, 0] = scale
        M[1, 1] = scale
        M[0, 2] = dst_center[0] - scale * src_center[0]
        M[1, 2] = dst_center[1] - scale * src_center[1]
        return M
