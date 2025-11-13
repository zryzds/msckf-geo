"""
Feature tracking module.

Tracks visual features across camera frames.
"""

import numpy as np  # 数值计算库
try:
    import cv2  # 可选：OpenCV，用于真实的 FAST/KLT 实现
    _CV_AVAILABLE = True
except Exception:
    cv2 = None
    _CV_AVAILABLE = False
from typing import List, Dict, Tuple, Optional  # 类型提示相关
from ..core.types import Feature, FeatureStatus  # 项目内通用特征数据结构及状态枚举
from collections import defaultdict  # Python 标准库：提供带默认值的 dict


class FeatureTracker:
    """
    特征轨迹管理器。

    该类仅维护 **特征轨迹**（Feature Track）及其观测，不负责从图像中提取/跟踪——
    具体的像素级跟踪交由 `OpticalFlowTracker` 等其他类完成。
    """

    def __init__(self,
                 max_track_length: int = 20,
                 min_track_length: int = 3,
                 max_features: int = 200):
        """
        初始化参数。

        Args:
            max_track_length: 同一特征可被连续跟踪的最大帧数，超出则视为可用于更新并移除。
            min_track_length: 特征最少被观测次数，少于该阈值则不用于状态更新。
            max_features: 同时维护的最大特征数量，超过后新特征将被忽略。
        """
        self.max_track_length = max_track_length  # 最长跟踪长度
        self.min_track_length = min_track_length  # 最小有效长度
        self.max_features = max_features          # 最大跟踪特征数

        self.active_tracks: Dict[int, Feature] = {}  # 当前存活的特征轨迹表：feature_id -> Feature 对象

        self._feature_id_counter = 0  # 生成唯一 feature_id 的自增计数器
        self._frame_id_counter = 0    # 记录帧编号（若需）

    def add_features(self, features: List[Tuple[float, float]], camera_state_id: int) -> List[Feature]:
        """
        将当前帧新检测到的特征加入跟踪器。

        Args:
            features: 特征点列表，每个元素为 (u, v) 归一化像素坐标。
            camera_state_id: 当前帧在 MSCKF 中对应的 CameraState ID。

        Returns:
            new_features: 新建的 Feature 对象列表。
        """
        new_features = []  # 存放返回结果

        for u, v in features:
            # 若已达到最大跟踪数，则停止添加
            if len(self.active_tracks) >= self.max_features:
                break

            # 创建 Feature 实例
            feature = Feature(
                feature_id=self._feature_id_counter,  # 分配唯一 ID
                u=u,
                v=v,
                camera_id=0  # 目前仅支持单目，因此固定为 0
            )
            # 记录第一次观测
            feature.add_observation(camera_state_id, np.array([u, v]))

            # 加入活动轨迹表
            self.active_tracks[self._feature_id_counter] = feature
            new_features.append(feature)

            # 自增 ID 计数器，为下一个特征做准备
            self._feature_id_counter += 1

        return new_features

    def track_features(self,
                      prev_features: List[Feature],
                      curr_observations: List[Tuple[int, float, float]],
                      camera_state_id: int) -> Tuple[List[Feature], List[Feature]]:
        """
        根据当前帧的观测结果更新上一帧特征状态。

        Args:
            prev_features: 上一帧跟踪到的 Feature 列表。
            curr_observations: 当前帧观测到的 (feature_id, u, v) 列表。
            camera_state_id: 当前帧的 CameraState ID。

        Returns:
            tracked: 成功继续跟踪的特征列表。
            lost:    在当前帧中丢失的特征列表。
        """
        tracked = []  # 成功跟踪的特征
        lost = []     # 丢失的特征

        # 将当前观测转换成 dict，便于 O(1) 查找
        curr_obs_dict = {fid: (u, v) for fid, u, v in curr_observations}

        for feature in prev_features:
            if feature.feature_id in curr_obs_dict:
                # 成功匹配到同一特征
                u, v = curr_obs_dict[feature.feature_id]
                feature.u = u  # 更新当前像素坐标
                feature.v = v
                feature.add_observation(camera_state_id, np.array([u, v]))

                # 若轨迹过长，则标记为可用于滤波更新
                if feature.num_observations >= self.max_track_length:
                    feature.status = FeatureStatus.READY_FOR_UPDATE
                else:
                    feature.status = FeatureStatus.TRACKED

                tracked.append(feature)
            else:
                # 在当前帧中未找到此特征 → 视为丢失
                feature.status = FeatureStatus.LOST
                lost.append(feature)

                # 从活动轨迹中移除
                if feature.feature_id in self.active_tracks:
                    del self.active_tracks[feature.feature_id]

        return tracked, lost

    def get_features_for_update(self) -> List[Feature]:
        """
        收集已满足条件、可用于 MSCKF 更新的特征。

        Returns:
            ready_features: 可用于状态更新的特征列表。
        """
        ready_features = []

        for feature in list(self.active_tracks.values()):
            # 满足条件：
            # 1) 主动标记 READY_FOR_UPDATE；或
            # 2) 已经丢失（LOST），需要边缘化。
            if (feature.status == FeatureStatus.READY_FOR_UPDATE or
                feature.status == FeatureStatus.LOST):

                # 至少达到最小观测次数，保证几何可观测性
                if feature.num_observations >= self.min_track_length:
                    ready_features.append(feature)
                    feature.status = FeatureStatus.MARGINALIZED  # 标记为已边缘化

                # 无论是否加入更新，都从活动轨迹中移除
                if feature.feature_id in self.active_tracks:
                    del self.active_tracks[feature.feature_id]

        return ready_features

    def prune_tracks(self, valid_camera_state_ids: List[int]):
        """
        清理与已删除 CameraState 关联的观测，防止内存泄漏。

        Args:
            valid_camera_state_ids: 仍然存在于滑动窗口中的 CameraState ID 列表。
        """
        valid_ids_set = set(valid_camera_state_ids)

        for feature in list(self.active_tracks.values()):
            # 过滤掉已经不存在的 CameraState 观测
            feature.observations = [
                (cam_id, obs) for cam_id, obs in feature.observations
                if cam_id in valid_ids_set
            ]

            # 若该特征已无任何观测，直接删除
            if len(feature.observations) == 0:
                if feature.feature_id in self.active_tracks:
                    del self.active_tracks[feature.feature_id]

    def remove_features_by_id(self, feature_ids: List[int]):
        """根据 ID 列表移除活跃特征（用于质量剪枝）。"""
        for fid in feature_ids:
            if fid in self.active_tracks:
                del self.active_tracks[fid]

    def reset(self):
        """重置跟踪器内部状态。"""
        self.active_tracks.clear()
        self._feature_id_counter = 0
        self._frame_id_counter = 0

    def get_active_features(self) -> List[Feature]:
        """返回当前仍在跟踪的全部特征对象列表。"""
        return list(self.active_tracks.values())

    @property
    def num_active_tracks(self) -> int:
        """当前活跃特征数量。"""
        return len(self.active_tracks)


class OpticalFlowTracker:
    """
    基于光流的特征跟踪器（占位实现）。

    实际项目中建议使用 OpenCV 的 `cv2.calcOpticalFlowPyrLK` 或其它更先进的
    跟踪算法；此处仅给出接口与最简示例，演示如何集成。
    """

    def __init__(self,
                 window_size: Tuple[int, int] = (21, 21),
                 max_level: int = 3,
                 min_eigen_threshold: float = 0.001,
                 lk_error_threshold: float = 12.0,
                 ransac_threshold: float = 3.0,
                 ransac_confidence: float = 0.99,
                 ransac_max_iterations: int = 1000):
        """
        初始化光流跟踪器。

        Args:
            window_size: 金字塔金字格中搜索窗口大小。
            max_level: 金字塔层级数。
            min_eigen_threshold: Shi-Tomasi 角点最小特征值阈值。
        """
        self.window_size = window_size
        self.max_level = max_level
        self.min_eigen_threshold = min_eigen_threshold
        # 质量控制参数
        self.lk_error_threshold = lk_error_threshold
        self.ransac_threshold = ransac_threshold
        self.ransac_confidence = ransac_confidence
        self.ransac_max_iterations = ransac_max_iterations

    def track(self,
             prev_image: np.ndarray,
             curr_image: np.ndarray,
             prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在两帧图像间跟踪点。

        Args:
            prev_image: 前一帧灰度图。
            curr_image: 当前帧灰度图。
            prev_points: 需要被跟踪的像素点 (N,2)。

        Returns:
            (curr_points, status)
            - curr_points: 跟踪后得到的像素坐标 (N,2)
            - status:      布尔数组，指示每个点是否成功跟踪
        """
        # 输入检查与预处理
        if prev_points is None or len(prev_points) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

        # 保证灰度图
        if prev_image.ndim == 3:
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY) if _CV_AVAILABLE else prev_image[:, :, 0]
        if curr_image.ndim == 3:
            curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY) if _CV_AVAILABLE else curr_image[:, :, 0]

        prev_pts = np.asarray(prev_points, dtype=np.float32).reshape(-1, 1, 2)

        if _CV_AVAILABLE:
            # 调用金字塔 LK 光流（KLT）
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(
                prev_image, curr_image, prev_pts, None,
                winSize=self.window_size,
                maxLevel=self.max_level,
                criteria=criteria,
                minEigThreshold=self.min_eigen_threshold
            )

            # 处理输出形状与状态
            curr_points = (next_pts.reshape(-1, 2) if next_pts is not None else np.zeros((len(prev_pts), 2), dtype=np.float32))
            st = st.reshape(-1).astype(bool)
            err = (err.reshape(-1) if err is not None else np.zeros((len(prev_pts),), dtype=np.float32))

            # 1) LK 误差阈值过滤（去除低置信度跟踪）
            st_lk = st & (err < self.lk_error_threshold)

            # 2) RANSAC 基础几何一致性过滤（去除外点）
            # 仅在足够点数时启用（≥8）
            if np.count_nonzero(st_lk) >= 8:
                prev_good = prev_pts.reshape(-1, 2)[st_lk]
                curr_good = curr_points[st_lk]
                try:
                    F, mask = cv2.findFundamentalMat(
                        prev_good, curr_good,
                        method=cv2.FM_RANSAC,
                        ransacReprojThreshold=self.ransac_threshold,
                        confidence=self.ransac_confidence,
                        maxIters=self.ransac_max_iterations
                    )
                    if mask is not None:
                        mask = mask.reshape(-1).astype(bool)
                        # 将 RANSAC 内点掩码映射回原索引
                        st_final = st_lk.copy()
                        # 遍历 st_lk 为 True 的位置，应用 mask
                        true_indices = np.flatnonzero(st_lk)
                        for local_i, idx in enumerate(true_indices):
                            if not bool(mask[local_i]):
                                st_final[idx] = False
                        st = st_final
                    else:
                        st = st_lk
                except Exception:
                    st = st_lk
            else:
                st = st_lk

            return curr_points, st
        else:
            # 无 OpenCV 时的退化实现：返回原坐标，全部视为成功
            curr_points = prev_pts.reshape(-1, 2).copy()
            status = np.ones(curr_points.shape[0], dtype=bool)
            return curr_points, status


class FeatureDetector:
    """
    特征检测器接口（占位实现）。

    在真实应用中，可对接 FAST、ORB、SIFT、Shi-Tomasi 等角点/特征检测算法。
    """

    def __init__(self,
                 detector_type: str = "FAST",
                 num_features: int = 200,
                 quality_level: float = 0.01,
                 min_distance: float = 10.0,
                 fast_threshold: int = 20):
        """
        初始化特征检测器。

        Args:
            detector_type: 检测器类型（FAST、ORB、SIFT 等）。
            num_features: 期望检测到的最大特征数量。
            quality_level: 角点质量阈值（如 Shi-Tomasi 检测中的阈值）。
            min_distance: 不同特征之间的最小像素间距。
        """
        self.detector_type = detector_type
        self.num_features = num_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.fast_threshold = fast_threshold

    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        在图像中检测特征点（占位实现）。

        Args:
            image: 输入灰度图像。
            mask:  (可选) 指定的检测区域掩码。

        Returns:
            points: 检测到的特征点数组 (N,2)。
        """
        # 保证灰度图
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if _CV_AVAILABLE else image[:, :, 0]

        if not _CV_AVAILABLE:
            # 无 OpenCV：退化为空结果，提醒在 README/requirements 中启用 CV 依赖
            return np.zeros((0, 2))

        # FAST 检测器
        if self.detector_type.upper() == "FAST":
            detector = cv2.FastFeatureDetector_create(threshold=self.fast_threshold, nonmaxSuppression=True)
            keypoints = detector.detect(image, mask)
            # 排序并截断到期望数量
            if len(keypoints) > self.num_features:
                # 尝试按响应排序（若有）
                try:
                    keypoints.sort(key=lambda k: k.response, reverse=True)
                except Exception:
                    pass
                keypoints = keypoints[:self.num_features]
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            return points.reshape(-1, 2)

        # 备用：Shi-Tomasi（KLT 友好）
        if self.detector_type.upper() in ("GFTT", "SHI-TOMASI", "KLT"):
            pts = cv2.goodFeaturesToTrack(
                image,
                maxCorners=self.num_features,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                mask=mask,
                useHarrisDetector=False
            )
            if pts is None:
                return np.zeros((0, 2))
            return pts.reshape(-1, 2).astype(np.float32)

        # 其它类型暂未实现：返回空
        return np.zeros((0, 2))

    def detect_grid(self,
                   image: np.ndarray,
                   grid_size: Tuple[int, int] = (5, 5),
                   features_per_cell: int = 5) -> np.ndarray:
        """
        在图像网格上均匀检测特征点。

        Args:
            image: 输入图像。
            grid_size: 网格划分 (行, 列)。
            features_per_cell: 每个网格 cell 内最多检测的特征数。

        Returns:
            points: 检测到的特征点数组 (N,2)。
        """
        height, width = image.shape[:2]  # 图像尺寸
        cell_height = height // grid_size[0]  # 每个网格的高
        cell_width = width // grid_size[1]    # 每个网格的宽

        all_points = []  # 收集所有网格的特征

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # 计算当前网格的像素范围
                y_start = i * cell_height
                y_end = (i + 1) * cell_height if i < grid_size[0] - 1 else height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width if j < grid_size[1] - 1 else width

                # 生成掩码，仅在当前网格区域内检测特征
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y_start:y_end, x_start:x_end] = 255

                # 调用 detect 检测
                cell_points = self.detect(image, mask)

                # 限制单个网格的特征数量
                if len(cell_points) > features_per_cell:
                    cell_points = cell_points[:features_per_cell]

                all_points.append(cell_points)

        if all_points:
            return np.vstack(all_points)  # 合并为一个 (N,2) 数组
        else:
            return np.zeros((0, 2))  # 若无特征则返回空
