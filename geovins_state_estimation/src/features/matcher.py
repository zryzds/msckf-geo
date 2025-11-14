"""
Feature matching module.

Provides feature matching algorithms for establishing correspondences.

新增：SuperPoint + LightGlue 深度特征匹配实现。
"""
# 导入数值计算库 NumPy，用于向量与矩阵运算
import numpy as np
# 导入类型提示：列表、元组、可选类型
from typing import List, Tuple, Optional
# 从核心类型模块导入 Feature 数据结构
from ..core.types import Feature

# 可选依赖：PyTorch 与 LightGlue
try:
    import torch  # 深度学习张量与推理
    from lightglue import LightGlue, SuperPoint  # 轻量化匹配器与特征提取器
    _LG_AVAILABLE = True
except Exception:
    torch = None
    LightGlue = None
    SuperPoint = None
    _LG_AVAILABLE = False

# 可选依赖：OpenCV 用于图像读写与可视化
try:
    import cv2
    _CV_AVAILABLE = True
except Exception:
    cv2 = None
    _CV_AVAILABLE = False


class FeatureMatcher:
    """
    Feature matcher using descriptor similarity.

    Supports various matching strategies including brute force and FLANN.
    """

    def __init__(self,
                 match_threshold: float = 0.7,
                 use_ratio_test: bool = True,
                 ratio_threshold: float = 0.8):
        """
        Initialize feature matcher.

        Args:
            match_threshold: Distance threshold for matches
            use_ratio_test: Use Lowe's ratio test
            ratio_threshold: Ratio for Lowe's ratio test
        """
        # 匹配距离阈值（越小越严格，影响匹配保留数量）
        self.match_threshold = match_threshold
        # 是否启用 Lowe 比率检验，提升匹配的鲁棒性
        self.use_ratio_test = use_ratio_test
        # Lowe 比率阈值（最佳/次佳的距离比值阈）
        self.ratio_threshold = ratio_threshold

    def match_features(self,
                      features1: List[Feature],
                      features2: List[Feature],
                      descriptors1: Optional[np.ndarray] = None,
                      descriptors2: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Match features between two sets.

        Args:
            features1: Features from first set
            features2: Features from second set
            descriptors1: Optional descriptors for features1
            descriptors2: Optional descriptors for features2

        Returns:
            List of matches (idx1, idx2)
        """
        # 若未提供描述子，则退回到基于像素坐标的最近邻匹配
        if descriptors1 is None or descriptors2 is None:
            # Fall back to coordinate-based matching
            return self._match_by_coordinates(features1, features2)

        # 存放输出的匹配对 (idx1, idx2)
        matches = []

        # 遍历第一组的每个描述子，与第二组全部描述子计算距离
        for i, desc1 in enumerate(descriptors1):
            # 计算到第二组所有描述子的欧氏距离
            distances = np.linalg.norm(descriptors2 - desc1, axis=1)

            # 找到最佳和次佳匹配（按距离从小到大排序）
            sorted_indices = np.argsort(distances)
            best_idx = sorted_indices[0]
            best_dist = distances[best_idx]

            # 若启用比率检验且存在次佳匹配
            if self.use_ratio_test and len(sorted_indices) > 1:
                second_best_dist = distances[sorted_indices[1]]  # 次佳距离
                ratio = best_dist / second_best_dist             # Lowe 比率

                # 同时满足比率阈值与距离阈值则接受匹配
                if ratio < self.ratio_threshold and best_dist < self.match_threshold:
                    matches.append((i, best_idx))
            else:
                # 未启用比率检验，仅使用距离阈值筛选
                if best_dist < self.match_threshold:
                    matches.append((i, best_idx))

        return matches

    def _match_by_coordinates(self,
                             features1: List[Feature],
                             features2: List[Feature],
                             max_distance: float = 20.0) -> List[Tuple[int, int]]:
        """
        Match features based on coordinate proximity.

        Args:
            features1: Features from first set
            features2: Features from second set
            max_distance: Maximum pixel distance for match

        Returns:
            List of matches (idx1, idx2)
        """
        # 基于坐标的近邻匹配结果列表
        matches = []

        # 遍历第一组特征，逐一在第二组中寻找最近点
        for i, f1 in enumerate(features1):
            min_dist = float('inf')       # 当前最小距离
            best_match = -1               # 当前最佳匹配索引

            for j, f2 in enumerate(features2):
                # 计算像素平面上的欧氏距离
                dist = np.sqrt((f1.u - f2.u)**2 + (f1.v - f2.v)**2)

                # 小于阈值且刷新最小距离时，记录为当前最佳匹配
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    best_match = j

            # 若找到合法匹配，则加入结果
            if best_match >= 0:
                matches.append((i, best_match))

        return matches


class EpipolarMatcher:
    """
    Feature matcher using epipolar geometry constraints.

    Matches features along epipolar lines between views.
    """

    def __init__(self,
                 epipolar_threshold: float = 1.0,
                 descriptor_threshold: float = 0.7):
        """
        Initialize epipolar matcher.

        Args:
            epipolar_threshold: Maximum distance from epipolar line (pixels)
            descriptor_threshold: Descriptor distance threshold
        """
        # 点到极线的最大允许距离（像素），用于候选集筛选
        self.epipolar_threshold = epipolar_threshold
        # 描述子距离阈值，用于在候选集中选出最佳匹配
        self.descriptor_threshold = descriptor_threshold

    def match_with_fundamental(self,
                               features1: List[Feature],
                               features2: List[Feature],
                               F: np.ndarray,
                               descriptors1: Optional[np.ndarray] = None,
                               descriptors2: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Match features using fundamental matrix constraint.

        Args:
            features1: Features from first image
            features2: Features from second image
            F: Fundamental matrix (3x3)
            descriptors1: Optional descriptors for features1
            descriptors2: Optional descriptors for features2

        Returns:
            List of matches (idx1, idx2)
        """
        # 返回的匹配索引列表
        matches = []

        for i, f1 in enumerate(features1):
            # 计算第一幅图中点 p1 在第二幅图中的极线 l
            p1 = np.array([f1.u, f1.v, 1.0])  # 齐次坐标
            epipolar_line = F @ p1            # l = F @ p1

            # Find features in image 2 close to epipolar line
            candidates = []  # 满足几何约束的候选索引

            for j, f2 in enumerate(features2):
                p2 = np.array([f2.u, f2.v, 1.0])  # 第二幅图中的点

                # 点到线距离：|ax + by + c| / sqrt(a^2 + b^2)
                distance = abs(epipolar_line @ p2) / np.sqrt(
                    epipolar_line[0]**2 + epipolar_line[1]**2
                )

                # 小于阈值的点作为候选
                if distance < self.epipolar_threshold:
                    candidates.append(j)

            # 在候选集中使用描述子进行匹配选择
            if len(candidates) > 0 and descriptors1 is not None and descriptors2 is not None:
                desc1 = descriptors1[i]                 # 当前点的描述子
                candidate_descs = descriptors2[candidates]  # 候选点描述子集合

                # 计算与候选集的距离并选择最近者
                distances = np.linalg.norm(candidate_descs - desc1, axis=1)
                best_idx = np.argmin(distances)

                # 小于阈值则接受匹配
                if distances[best_idx] < self.descriptor_threshold:
                    matches.append((i, candidates[best_idx]))

        return matches

    @staticmethod
    def compute_fundamental_matrix(
        points1: np.ndarray,
        points2: np.ndarray,
        method: str = "8point"
    ) -> Optional[np.ndarray]:
        """
        Compute fundamental matrix from point correspondences.

        Args:
            points1: Points in first image (N, 2)
            points2: Points in second image (N, 2)
            method: Method to use ("8point" or "ransac")

        Returns:
            Fundamental matrix (3x3) or None if computation fails
        """
        # 八点法至少需要 8 对点
        if len(points1) < 8:
            return None

        # 点归一化以提升数值稳定性
        points1_norm, T1 = EpipolarMatcher._normalize_points(points1)
        points2_norm, T2 = EpipolarMatcher._normalize_points(points2)

        # 构造线性约束矩阵 A
        N = len(points1)
        A = np.zeros((N, 9))

        for i in range(N):
            x1, y1 = points1_norm[i]
            x2, y2 = points2_norm[i]

            # 每对点提供一行线性约束
            A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

        # 通过 SVD 求解最小二乘解（最后一行的奇异向量）
        _, _, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)

        # 强制秩为 2（将最小奇异值置零）
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ Vt

        # 将 F 反变换回原始像素坐标系
        F = T2.T @ F @ T1

        return F

    @staticmethod
    def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize points for numerical stability.

        Args:
            points: Points (N, 2)

        Returns:
            (normalized_points, transformation_matrix)
        """
        # 计算质心并平移点使其以质心为中心
        centroid = np.mean(points, axis=0)
        shifted = points - centroid

        # 计算平均距离以决定缩放，使均值为 sqrt(2)
        mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
        scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0

        # 构造归一化变换矩阵 T
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])

        # 将原始点转为齐次坐标并应用 T 进行归一化
        points_homog = np.column_stack([points, np.ones(len(points))])
        points_norm = (T @ points_homog.T).T
        points_norm = points_norm[:, :2] / points_norm[:, 2:3]

        return points_norm, T


class RANSACMatcher:
    """
    RANSAC-based feature matcher for outlier rejection.
    """

    def __init__(self,
                 ransac_threshold: float = 3.0,
                 max_iterations: int = 1000,
                 confidence: float = 0.99):
        """
        Initialize RANSAC matcher.

        Args:
            ransac_threshold: Inlier threshold (pixels)
            max_iterations: Maximum RANSAC iterations
            confidence: Desired confidence level
        """
        # 内点阈值（像素单位），用于判断匹配是否符合模型
        self.ransac_threshold = ransac_threshold
        # 最大迭代次数
        self.max_iterations = max_iterations
        # 期望的置信度，用于估算提前终止条件
        self.confidence = confidence


class SuperPointLightGlueMatcher:
    """
    基于 SuperPoint + LightGlue 的深度特征匹配器。

    - SuperPoint 负责在图像上检测关键点并提取描述子。
    - LightGlue 负责根据两幅图的关键点与描述子进行高效鲁棒的匹配。

    使用示例：
        matcher = SuperPointLightGlueMatcher()
        result = matcher.match_paths(path0, path1)
        matcher.visualize(result, save_path)

    返回结果字典包含：
        - keypoints0 (np.ndarray, Nx2)
        - keypoints1 (np.ndarray, Nx2)
        - matches_idx (np.ndarray, Kx2)
        - scores (np.ndarray, K)
        - image0 / image1 (np.ndarray, HxWxC, 可视化用途)
    """

    def __init__(self,
                 device: Optional[str] = None,
                 max_num_keypoints: Optional[int] = 2048,
                 depth_confidence: float = 0.9,
                 width_confidence: float = 0.95):
        if not _LG_AVAILABLE:
            raise ImportError(
                "LightGlue/SuperPoint 未安装。请先安装: pip install torch torchvision opencv-python git+https://github.com/cvg/LightGlue.git"
            )

        # 选择设备：优先 CUDA，其次 MPS，最后 CPU
        if device is not None:
            self.device = device
        else:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # 初始化 SuperPoint 特征提取器与 LightGlue 匹配器
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint',
                                  depth_confidence=depth_confidence,
                                  width_confidence=width_confidence).eval().to(self.device)

    @staticmethod
    def _ensure_gray(image: np.ndarray) -> np.ndarray:
        """将输入图像转换为灰度并返回 float32 数组，值域 [0, 1]。"""
        if image.ndim == 3:
            # 假设为 BGR（三通道），转换为灰度
            if _CV_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                # 无 OpenCV 时的简易转换：按通道平均
                gray = image.astype(np.float32).mean(axis=2)
        else:
            gray = image

        gray = gray.astype(np.float32)
        # 若像素值为 0-255，则归一化到 [0, 1]
        if gray.max() > 1.0:
            gray = gray / 255.0
        return gray

    def _to_tensor(self, image: np.ndarray) -> 'torch.Tensor':
        """将灰度图转换为 (1,1,H,W) 的 torch 张量并放置到设备。"""
        assert _LG_AVAILABLE, "需要安装 torch 与 LightGlue/SuperPoint"
        gray = self._ensure_gray(image)
        h, w = gray.shape[:2]
        t = torch.from_numpy(gray)[None, None, :, :].to(self.device)
        return t

    def extract(self, image: np.ndarray) -> dict:
        """对单幅图像执行 SuperPoint 特征提取。"""
        t = self._to_tensor(image)
        with torch.inference_mode():
            feats = self.extractor({'image': t})
        return feats

    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        """在两幅图像之间执行 SuperPoint+LightGlue 匹配。"""
        # 特征提取
        feats0 = self.extract(img0)
        feats1 = self.extract(img1)

        # 图像尺寸信息，LightGlue 预期顺序为 [W, H]
        size0 = torch.tensor([[img0.shape[1], img0.shape[0]]], dtype=torch.float32, device=self.device)
        size1 = torch.tensor([[img1.shape[1], img1.shape[0]]], dtype=torch.float32, device=self.device)

        # 调用 LightGlue 前向匹配
        with torch.inference_mode():
            matches01 = self.matcher({
                'image0': {
                    'keypoints': feats0['keypoints'],
                    'descriptors': feats0['descriptors'],
                    'image_size': size0,
                },
                'image1': {
                    'keypoints': feats1['keypoints'],
                    'descriptors': feats1['descriptors'],
                    'image_size': size1,
                }
            })

        # 提取匹配索引与得分
        matches = matches01.get('matches', None)
        scores = matches01.get('scores', None)
        if matches is None:
            # 兼容：如果没有紧凑格式，则尝试从 matches0/matches1 还原
            m0 = matches01.get('matches0', None)
            if m0 is None:
                raise RuntimeError('LightGlue 未返回匹配结果。')
            # 从稠密格式恢复匹配对（只处理 batch=1 情况）
            if isinstance(m0, torch.Tensor):
                m0_b0 = m0[0] if m0.dim() > 1 else m0
            else:
                # list 情况
                m0_b0 = m0[0]
            valid = (m0_b0 > -1)
            idx0 = torch.where(valid)[0]
            idx1 = m0_b0[valid]
            matches = [torch.stack([idx0, idx1], dim=-1)]
            # 若提供 matching_scores0，则一并取出
            s0 = matches01.get('matching_scores0', None)
            if isinstance(s0, torch.Tensor):
                s0_b0 = s0[0] if s0.dim() > 1 else s0
                scores = [s0_b0[valid]]
            else:
                scores = None

        # 去除 batch 维度（兼容 list / tensor）
        def _rm_batch(x):
            if x is None:
                return None
            if isinstance(x, list):
                x = x[0]
            elif isinstance(x, torch.Tensor) and x.dim() > 2 and x.shape[0] == 1:
                x = x[0]
            return x

        kpts0 = _rm_batch(feats0['keypoints'])
        kpts1 = _rm_batch(feats1['keypoints'])
        matches = _rm_batch(matches)
        scores = _rm_batch(scores)

        # 转换为 numpy
        kpts0 = kpts0.detach().cpu().numpy()
        kpts1 = kpts1.detach().cpu().numpy()
        matches_np = matches.detach().cpu().numpy()
        scores_np = (scores.detach().cpu().numpy() if scores is not None else None)

        # 生成每对匹配的坐标
        mkpts0 = kpts0[matches_np[:, 0]]
        mkpts1 = kpts1[matches_np[:, 1]]

        return {
            'keypoints0': mkpts0,
            'keypoints1': mkpts1,
            'matches_idx': matches_np,
            'scores': scores_np,
            'image0': img0,
            'image1': img1,
        }

    def match_paths(self, path0: str, path1: str) -> dict:
        """读取两幅图像路径并执行匹配。"""
        if not _CV_AVAILABLE:
            raise ImportError('OpenCV 未安装，无法读取图像。请安装 opencv-python')
        img0 = cv2.imread(path0, cv2.IMREAD_COLOR)
        img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        if img0 is None or img1 is None:
            raise FileNotFoundError(f'无法读取图像: {path0} 或 {path1}')
        return self.match(img0, img1)

    def visualize(self, result: dict, save_path: Optional[str] = None,
                  color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1) -> np.ndarray:
        """根据匹配结果进行可视化并保存。

        可视化策略：并排显示两幅图像，连线绘制匹配对。
        """
        if not _CV_AVAILABLE:
            raise ImportError('OpenCV 未安装，无法进行可视化。请安装 opencv-python')

        img0 = result['image0']
        img1 = result['image1']
        mkpts0 = result['keypoints0']
        mkpts1 = result['keypoints1']

        # 将两张图横向拼接
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        H = max(h0, h1)
        W = w0 + w1

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:h0, :w0] = img0
        canvas[:h1, w0:w0 + w1] = img1

        # 绘制匹配连线与关键点
        for (p0, p1) in zip(mkpts0, mkpts1):
            x0, y0 = int(round(p0[0])), int(round(p0[1]))
            x1, y1 = int(round(p1[0])) + w0, int(round(p1[1]))
            cv2.line(canvas, (x0, y0), (x1, y1), color, thickness)
            cv2.circle(canvas, (x0, y0), 2, (0, 0, 255), -1)
            cv2.circle(canvas, (x1, y1), 2, (255, 0, 0), -1)

        if save_path:
            cv2.imwrite(save_path, canvas)
        return canvas

    def refine_matches(self,
                      matches: List[Tuple[int, int]],
                      features1: List[Feature],
                      features2: List[Feature]) -> List[Tuple[int, int]]:
        """
        Refine matches using RANSAC.

        Args:
            matches: Initial matches
            features1: Features from first image
            features2: Features from second image

        Returns:
            Filtered matches (inliers only)
        """
        # 至少需要 8 对匹配点才能估计基础矩阵
        if len(matches) < 8:
            return matches

        # 提取像素坐标用于估计（按匹配索引聚合）
        points1 = np.array([[features1[m[0]].u, features1[m[0]].v] for m in matches])
        points2 = np.array([[features2[m[1]].u, features2[m[1]].v] for m in matches])

        # RANSAC 主循环，记录当前最佳内点集与对应基础矩阵
        best_inliers = []
        best_F = None

        for _ in range(self.max_iterations):
            # 随机采样 8 对匹配点用于估计基础矩阵
            sample_indices = np.random.choice(len(matches), 8, replace=False)
            sample_points1 = points1[sample_indices]
            sample_points2 = points2[sample_indices]

            # 计算基础矩阵 F（八点法）
            F = EpipolarMatcher.compute_fundamental_matrix(
                sample_points1,
                sample_points2,
                method="8point"
            )

            # 若估计失败则跳过本次迭代
            if F is None:
                continue

            # 根据当前 F 计算内点集合
            inliers = self._compute_inliers(points1, points2, F)

            # 如果当前内点数更多，则更新最佳结果
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F

                # 早停检查：当内点比例足够高时可提前终止
                inlier_ratio = len(inliers) / len(matches)
                if self._check_termination(inlier_ratio, len(inliers)):
                    break

        # 返回内点匹配集合；若未找到更好的模型则返回原匹配
        if len(best_inliers) > 0:
            return [matches[i] for i in best_inliers]
        else:
            return matches

    def _compute_inliers(self,
                        points1: np.ndarray,
                        points2: np.ndarray,
                        F: np.ndarray) -> List[int]:
        """Compute inliers for given fundamental matrix."""
        inliers = []  # 记录满足误差阈值的匹配索引

        for i, (p1, p2) in enumerate(zip(points1, points2)):
            # 转为齐次坐标
            p1_h = np.array([p1[0], p1[1], 1.0])
            p2_h = np.array([p2[0], p2[1], 1.0])

            # 对称极线距离（更稳定的误差度量）
            error1 = abs(p2_h @ F @ p1_h) / np.sqrt((F @ p1_h)[0]**2 + (F @ p1_h)[1]**2)
            error2 = abs(p1_h @ F.T @ p2_h) / np.sqrt((F.T @ p2_h)[0]**2 + (F.T @ p2_h)[1]**2)

            error = (error1 + error2) / 2

            # 误差小于阈值则视为内点
            if error < self.ransac_threshold:
                inliers.append(i)

        return inliers

    def _check_termination(self, inlier_ratio: float, num_inliers: int) -> bool:
        """Check if RANSAC should terminate early."""
        # 小于 8 个内点不足以稳定估计，不能提前终止
        if num_inliers < 8:
            return False

        # 根据当前内点比例估计达到期望置信度所需迭代次数
        epsilon = 1 - inlier_ratio
        if epsilon >= 1:
            return False

        # 使用经典 RANSAC 公式估计迭代次数（模型需要 8 点）
        num_iterations_needed = np.log(1 - self.confidence) / np.log(1 - (1 - epsilon)**8)

        # 若最大迭代次数已超过所需次数，则可以提前终止
        return self.max_iterations > num_iterations_needed
