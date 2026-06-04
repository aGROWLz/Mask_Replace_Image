import torch
import numpy as np
import cv2
from .utils import get_mask_bounding_box


class MaskBoundingBox:
    """从遮罩提取边界框坐标"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    
    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("left", "top", "right", "bottom")
    
    def main(self, mask):
        """
        提取遮罩的边界框
        
        Args:
            mask: 输入遮罩
            
        Returns:
            (left, top, right, bottom) 边界框坐标
        """
        left, top, right, bottom = get_mask_bounding_box(mask)
        return (left, top, right, bottom)


class MergeMasks:
    """合并多个遮罩为一个遮罩"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }
    
    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("merged_mask",)
    
    def main(self, masks):
        """
        合并多个遮罩为一个遮罩
        
        Args:
            masks: 批量遮罩，shape为 (N, H, W) 或 (N, 1, H, W)
            
        Returns:
            合并后的遮罩 shape (1, H, W)
        """
        # 确保masks是tensor
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)
        
        # 处理不同的输入格式
        if masks.dim() == 2:
            # (H, W) -> (1, H, W)
            masks = masks.unsqueeze(0)
        elif masks.dim() == 4:
            # (N, 1, H, W) -> (N, H, W)
            masks = masks.squeeze(1)
        
        # 现在masks应该是 (N, H, W) 格式
        if masks.dim() != 3:
            raise ValueError(f"遮罩格式不正确，期望 (N, H, W)，得到 {masks.shape}")
        
        # 合并遮罩：对所有遮罩取最大值（取并集）
        # 这样可以保留所有物体的遮罩区域
        merged_mask = torch.max(masks, dim=0)[0]  # 对第一个维度取最大值
        
        # 确保输出格式为 (1, H, W)
        if merged_mask.dim() == 2:
            merged_mask = merged_mask.unsqueeze(0)
        
        return (merged_mask,)


class MergeMasksV2:
    """支持多输入端及批量遮罩的合并节点"""

    OPTIONAL_MASK_SLOTS = 8  # mask_2 ~ mask_9

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"mask_{idx}": ("MASK",)
            for idx in range(2, cls.OPTIONAL_MASK_SLOTS + 2)
        }
        optional_inputs["batched_masks"] = ("MASK",)
        return {
            "required": {
                "mask_1": ("MASK",),
            },
            "optional": optional_inputs,
        }

    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("merged_mask",)

    def main(
        self,
        mask_1,
        mask_2=None,
        mask_3=None,
        mask_4=None,
        mask_5=None,
        mask_6=None,
        mask_7=None,
        mask_8=None,
        mask_9=None,
        batched_masks=None,
    ):
        single_masks = []

        def collect(mask):
            if mask is None:
                return
            tensor = self._ensure_mask_tensor(mask)
            if tensor.shape[0] == 1:
                single_masks.append(tensor)
            else:
                for i in range(tensor.shape[0]):
                    single_masks.append(tensor[i : i + 1])

        for m in [
            mask_1,
            mask_2,
            mask_3,
            mask_4,
            mask_5,
            mask_6,
            mask_7,
            mask_8,
            mask_9,
        ]:
            collect(m)

        if batched_masks is not None:
            batch_tensor = self._ensure_mask_tensor(batched_masks)
            for i in range(batch_tensor.shape[0]):
                single_masks.append(batch_tensor[i : i + 1])

        if not single_masks:
            raise ValueError("至少需要提供一个有效的遮罩用于合并。")

        reference_shape = single_masks[0].shape
        for tensor in single_masks:
            if tensor.shape != reference_shape:
                raise ValueError(
                    f"所有遮罩尺寸需一致。参考尺寸为 {reference_shape}，收到 {tensor.shape}"
                )

        masks_tensor = torch.cat(single_masks, dim=0)  # (N, H, W)
        merged_mask = torch.max(masks_tensor, dim=0)[0]
        if merged_mask.dim() == 2:
            merged_mask = merged_mask.unsqueeze(0)

        return (merged_mask,)

    def _ensure_mask_tensor(self, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        if mask.dim() == 4:
            # (N, 1, H, W) -> (N, H, W)
            mask = mask.squeeze(1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"遮罩格式不正确，期望 (N, H, W)，得到 {mask.shape}")

        return mask.to(dtype=torch.float32)


class SelectLargestMask:
    """根据boxes面积筛选出最大的遮罩"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "boxes": ("STRING", {
                    "default": "[]",
                    "multiline": True
                }),
            }
        }
    
    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("largest_mask", "index")
    
    def main(self, masks, boxes):
        """
        根据boxes面积筛选出最大的遮罩
        
        Args:
            masks: 批量遮罩，shape为 (N, H, W) 或 (N, 1, H, W)
            boxes: boxes字符串或列表，格式如 "[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]"
                  或 [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            
        Returns:
            largest_mask: 最大的遮罩 shape (1, H, W)
            index: 最大遮罩的索引
        """
        import json
        import ast
        
        # 确保masks是tensor
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)
        
        # 处理不同的输入格式
        if masks.dim() == 2:
            # (H, W) -> (1, H, W)
            masks = masks.unsqueeze(0)
        elif masks.dim() == 4:
            # (N, 1, H, W) -> (N, H, W)
            masks = masks.squeeze(1)
        
        # 现在masks应该是 (N, H, W) 格式
        if masks.dim() != 3:
            raise ValueError(f"遮罩格式不正确，期望 (N, H, W)，得到 {masks.shape}")
        
        num_masks = masks.shape[0]
        
        # 解析boxes
        boxes_list = None
        try:
            if isinstance(boxes, str):
                # 尝试解析JSON字符串
                try:
                    boxes_list = json.loads(boxes)
                except:
                    # JSON解析失败，尝试作为Python字面量（更安全的方式）
                    try:
                        # 使用ast.literal_eval替代eval，更安全
                        boxes_list = ast.literal_eval(boxes)
                    except:
                        boxes_list = None
            elif isinstance(boxes, (list, tuple)):
                boxes_list = boxes
            elif isinstance(boxes, torch.Tensor):
                # 如果是tensor，转换为列表
                boxes_list = boxes.cpu().tolist()
            
            # 处理嵌套列表格式：[[[x1,y1,x2,y2], ...], ...] -> [[x1,y1,x2,y2], ...]
            if boxes_list and len(boxes_list) > 0:
                if isinstance(boxes_list[0], list) and len(boxes_list[0]) > 0:
                    if isinstance(boxes_list[0][0], list) or isinstance(boxes_list[0][0], (int, float)):
                        # 检查是否是嵌套格式
                        first_item = boxes_list[0]
                        if isinstance(first_item, list) and len(first_item) > 0:
                            if isinstance(first_item[0], list):
                                # 是嵌套格式 [[[x1,y1,x2,y2], ...], ...]，取第一层
                                boxes_list = boxes_list[0] if boxes_list else []
            
        except Exception as e:
            boxes_list = None
        
        # 计算每个box的面积
        areas = []
        if boxes_list and len(boxes_list) > 0:
            for box in boxes_list:
                try:
                    # 处理不同的box格式
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        # [x1, y1, x2, y2] 格式
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        area = abs((x2 - x1) * (y2 - y1))
                        areas.append(area)
                    elif isinstance(box, torch.Tensor) and box.numel() >= 4:
                        # tensor格式
                        box_values = box.cpu().tolist()
                        if isinstance(box_values, list):
                            x1, y1, x2, y2 = float(box_values[0]), float(box_values[1]), float(box_values[2]), float(box_values[3])
                        else:
                            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        area = abs((x2 - x1) * (y2 - y1))
                        areas.append(area)
                    else:
                        areas.append(0)
                except:
                    areas.append(0)
        
        # 如果没有有效的boxes，或者boxes数量与masks不匹配，使用遮罩像素数
        if not areas or len(areas) != num_masks:
            # 计算每个遮罩的非零像素数（面积）
            areas = []
            for i in range(num_masks):
                mask_area = torch.sum(masks[i] > 0).item()
                areas.append(mask_area)
        
        # 找到面积最大的索引
        if not areas or num_masks == 0:
            # 如果没有遮罩或没有有效的面积数据，返回空遮罩
            return (torch.zeros(1, 64, 64), -1)
        
        largest_idx = int(np.argmax(areas))
        
        # 提取最大的遮罩
        largest_mask = masks[largest_idx].unsqueeze(0)  # (H, W) -> (1, H, W)
        
        return (largest_mask, largest_idx)


class SelectLargestMaskByArea:
    """根据mask面积直接筛选出最大的遮罩（不依赖bbox）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }
    
    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("largest_mask", "index")
    
    def main(self, masks):
        """
        按连通域面积筛选最大的遮罩（阈值二值化 + 最大连通域，类似 Mask-filter）
        
        Args:
            masks: 批量遮罩，shape为 (N, H, W) 或 (N, 1, H, W) 或 (H, W)
        """
        # 确保masks是tensor
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)
        
        # 处理不同的输入格式
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)          # (H, W) -> (1, H, W)
        elif masks.dim() == 4:
            masks = masks.squeeze(1)            # (N, 1, H, W) -> (N, H, W)
        
        if masks.dim() != 3:
            raise ValueError(f"遮罩格式不正确，期望 (N, H, W)，得到 {masks.shape}")
        
        # 检查是否有遮罩
        if masks.shape[0] == 0:
            # 如果没有遮罩，返回空遮罩
            return (torch.zeros(1, 64, 64), -1)
        
        # 若只有单个遮罩，直接用连通域筛一遍
        if masks.shape[0] == 1:
            largest_mask = self._largest_component(masks[0])
            return (largest_mask.unsqueeze(0), 0)
        
        # 遍历批次，取每个mask的最大连通域面积，再整体取最大
        best_idx = 0
        best_area = -1
        best_mask = None
        for i in range(masks.shape[0]):
            lm = self._largest_component(masks[i])
            area = lm.sum().item()
            if area > best_area:
                best_area = area
                best_idx = i
                best_mask = lm
        
        if best_mask is None:
            # fallback：返回第一个
            return (masks[0].unsqueeze(0), 0)
        
        return (best_mask.unsqueeze(0), best_idx)
    
    def _largest_component(self, mask_tensor, threshold: float = 0.5):
        """
        对单个遮罩取最大连通域，返回 (H, W) tensor
        """
        mask_np = mask_tensor.cpu().numpy().astype(np.float32)
        if mask_np.ndim > 2:
            mask_np = mask_np[0]
        
        # 二值化
        binary = (mask_np >= threshold).astype(np.uint8)
        
        # 找连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return torch.from_numpy(binary.astype(np.float32))
        
        # 最大前景（排除背景label 0）
        max_label = 1
        max_area = stats[1, cv2.CC_STAT_AREA]
        for lbl in range(2, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = lbl
        
        largest = (labels == max_label).astype(np.float32)
        return torch.from_numpy(largest)


class MergeMasksDelete:
    """合并遮罩（删除）：从主遮罩中删除指定区域"""

    OPTIONAL_MASK_SLOTS = 9  # mask_del_1 ~ mask_del_9

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"mask_del_{idx}": ("MASK",)
            for idx in range(1, cls.OPTIONAL_MASK_SLOTS + 1)
        }
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": optional_inputs,
        }

    CATEGORY = "mask"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("result_mask",)

    def main(
        self,
        mask,
        mask_del_1=None,
        mask_del_2=None,
        mask_del_3=None,
        mask_del_4=None,
        mask_del_5=None,
        mask_del_6=None,
        mask_del_7=None,
        mask_del_8=None,
        mask_del_9=None,
    ):
        # 处理主遮罩
        main_mask = self._ensure_3d(mask)

        # 收集所有删除遮罩并取并集
        del_masks = []
        for m in [mask_del_1, mask_del_2, mask_del_3, mask_del_4,
                   mask_del_5, mask_del_6, mask_del_7, mask_del_8, mask_del_9]:
            if m is not None:
                t = self._ensure_3d(m)
                # 支持批量删除遮罩，逐个拆分
                for i in range(t.shape[0]):
                    del_masks.append(t[i:i+1])

        result = main_mask
        if del_masks:
            # 将所有删除遮罩合并（取并集）
            del_tensor = torch.cat(del_masks, dim=0)  # (N, H, W)
            del_union = torch.max(del_tensor, dim=0)[0]  # (H, W)
            del_union = del_union.unsqueeze(0)  # (1, H, W)

            # 检查尺寸是否匹配
            if del_union.shape[1:] != main_mask.shape[1:]:
                raise ValueError(
                    f"删除遮罩尺寸 {del_union.shape} 与主遮罩尺寸 {main_mask.shape} 不匹配"
                )

            # 从主遮罩中减去删除区域
            result = torch.clamp(main_mask - del_union, 0.0, 1.0)

        return (result,)

    def _ensure_3d(self, mask):
        """确保遮罩为 (N, H, W) 格式"""
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:
            mask = mask.squeeze(1)
        if mask.dim() != 3:
            raise ValueError(f"遮罩格式不正确，期望 (N, H, W)，得到 {mask.shape}")
        return mask.to(dtype=torch.float32)
