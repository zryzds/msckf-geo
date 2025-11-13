"""
Multi-State Constraint Kalman Filter (MSCKF) implementation.

MSCKF maintains a sliding window of camera poses and performs measurement updates
when features are marginalized out.
"""
import numpy as np
from typing import List, Tuple, Optional
from ..core.state import MSCKFState, IMUState, CameraState
from ..core.types import IMUMeasurement, Feature, Quaternion, Pose
from ..sensors.imu_model import IMUModel
from ..sensors.camera_model import PinholeCameraModel, Triangulation
from .ekf import ExtendedKalmanFilter


class MSCKF:
    """
    Multi-State Constraint Kalman Filter for visual-inertial navigation.

    Combines IMU measurements with visual feature observations to estimate
    vehicle trajectory and camera poses.
    """

    def __init__(self,
                 initial_state: MSCKFState,
                 imu_model: IMUModel,
                 camera_model: PinholeCameraModel,
                 T_cam_imu: Pose,
                 initial_covariance: Optional[np.ndarray] = None,
                 max_camera_states: int = 20):
        """
        Initialize MSCKF.

        Args:
            initial_state: Initial MSCKF state
            imu_model: IMU sensor model
            camera_model: Camera model for projection
            T_cam_imu: Extrinsic calibration (IMU to camera)
            initial_covariance: Initial state covariance
            max_camera_states: Maximum camera poses in sliding window
        """
        self.state = initial_state
        self.imu_model = imu_model
        self.camera_model = camera_model
        self.T_cam_imu = T_cam_imu

        # EKF for covariance
        if initial_covariance is None:
            initial_covariance = self._create_initial_covariance()

        self.ekf = ExtendedKalmanFilter(
            state_dim=self.state.state_dim,
            initial_covariance=initial_covariance
        )

        # Parameters
        self.max_camera_states = max_camera_states
        self.camera_state_counter = 0

        # Noise parameters
        self.feature_noise = 1.0  # pixel

    def propagate_imu(self, imu_measurements: List[IMUMeasurement]):
        """
        Propagate state using IMU measurements.

        Args:
            imu_measurements: List of IMU measurements
        """
        if len(imu_measurements) < 2:
            return

        # Propagate nominal state
        self.state.imu_state = self.imu_model.propagate(
            self.state.imu_state,
            imu_measurements
        )

        # Propagate covariance
        for i in range(len(imu_measurements) - 1):
            dt = imu_measurements[i + 1].timestamp - imu_measurements[i].timestamp
            if dt <= 0:
                continue

            omega = imu_measurements[i].angular_velocity - self.state.imu_state.gyro_bias
            accel = imu_measurements[i].linear_acceleration - self.state.imu_state.accel_bias

            # Compute state transition matrix
            F = self.imu_model.compute_state_transition(
                self.state.imu_state,
                omega,
                accel,
                dt
            )

            # Compute process noise
            Q = self.imu_model.compute_process_noise(dt)

            # Expand F and Q to full state dimension
            F_full = np.eye(self.state.state_dim)
            F_full[:15, :15] = F

            Q_full = np.zeros((self.state.state_dim, self.state.state_dim))
            Q_full[:15, :15] = Q

            # EKF prediction
            self.ekf.predict(F_full, Q_full)

    def augment_state(self, timestamp: float):
        """
        Augment state with new camera pose.

        Args:
            timestamp: Timestamp of camera frame
        """
        # If sliding window is full, marginalize the oldest camera state BEFORE adding new one,
        # so covariance and state dimensions remain consistent with MSCKFState.
        if self.state.num_camera_states >= self.state.max_camera_states and self.state.num_camera_states > 0:
            oldest_id = min(self.state.camera_states.keys())
            self.marginalize_camera_state(oldest_id)

        # Compute camera pose from IMU pose
        R_imu = self.state.imu_state.rotation_matrix
        p_imu = self.state.imu_state.position

        R_cam_imu = self.T_cam_imu.orientation.to_rotation_matrix()
        p_cam_imu = self.T_cam_imu.position

        # Camera pose in world frame
        R_cam = R_imu @ R_cam_imu
        p_cam = p_imu + R_imu @ p_cam_imu

        # Create camera state
        camera_state = CameraState(
            state_id=self.camera_state_counter,
            timestamp=timestamp,
            orientation=Quaternion.from_rotation_matrix(R_cam),
            position=p_cam
        )

        self.camera_state_counter += 1

        # Add to state
        self.state.add_camera_state(camera_state)

        # Augment covariance
        self._augment_covariance(camera_state)

    def _augment_covariance(self, camera_state: CameraState):
        """
        Augment covariance matrix with new camera state.

        Args:
            camera_state: New camera state
        """
        # Jacobian of camera pose w.r.t. IMU state
        R_imu = self.state.imu_state.rotation_matrix
        R_cam_imu = self.T_cam_imu.orientation.to_rotation_matrix()
        p_cam_imu = self.T_cam_imu.position

        # Position Jacobian
        J_p = np.zeros((6, 15))
        J_p[0:3, 0:3] = -R_imu @ self._skew_symmetric(R_cam_imu.T @ p_cam_imu)
        J_p[0:3, 3:6] = np.eye(3)
        J_p[3:6, 0:3] = np.eye(3)

        # New camera covariance
        P_imu = self.ekf.P[:15, :15]
        P_cam = J_p @ P_imu @ J_p.T

        # Cross-covariance
        P_cross = self.ekf.P[:, :15] @ J_p.T

        # Augment
        old_dim = self.ekf.state_dim
        new_dim = old_dim + 6

        P_aug = np.zeros((new_dim, new_dim))
        P_aug[:old_dim, :old_dim] = self.ekf.P
        P_aug[old_dim:, old_dim:] = P_cam
        P_aug[:old_dim, old_dim:] = P_cross
        P_aug[old_dim:, :old_dim] = P_cross.T

        self.ekf.P = P_aug
        self.ekf.state_dim = new_dim

    def update_features(self, features: List[Feature]) -> List[int]:
        """
        Perform measurement update using features.

        Args:
            features: List of features ready for update
        Returns:
            bad_feature_ids: 特征质量不合格（观测不足、三角化失败、负深度或重投影误差大）建议剪枝的 ID 列表。
        """
        if len(features) == 0:
            return []

        # Build measurement model for all features
        H_all = []
        r_all = []
        bad_feature_ids: List[int] = []

        for feature in features:
            H_f, r_f = self._compute_feature_jacobian(feature)

            if H_f is not None and r_f is not None:
                H_all.append(H_f)
                r_all.append(r_f)
            else:
                # 记录为待剪枝
                bad_feature_ids.append(feature.feature_id)

        if len(H_all) == 0:
            return bad_feature_ids

        # Stack measurements
        H = np.vstack(H_all)
        r = np.concatenate(r_all)

        # Measurement noise
        num_measurements = len(r)
        R = self.feature_noise**2 * np.eye(num_measurements)

        # EKF update
        state_correction, _ = self.ekf.update(H, R, r)

        # Apply state correction
        self._apply_state_correction(state_correction)

        return bad_feature_ids

    def _compute_feature_jacobian(self, feature: Feature) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        计算特征的测量雅可比（多视角堆叠一次性消元）。

        - 将同一特征的所有观测按 2×M 行块堆叠，形成 H_f_stack ∈ R^{2M×3}
        - 计算 H_f_stack 的左零空间 L ∈ R^{2M×(2M−rank)}（通常 rank=3 → 维度 2M−3）
        - 用 L^T 同时消元特征坐标，将残差与对各相机状态的雅可比统一投影到更强的约束上

        若三角化失败、观测不足或几何不良（负深度/重投影误差过大），则返回 (None, None)。
        """
        # 至少需要 3 次观测保证几何可观测性
        if feature.num_observations < 3:
            return None, None

        # 收集滑窗中对应的 CameraState 与归一化观测
        camera_states = []
        observations = []

        for cam_state_id, obs in feature.observations:
            cam_state = self.state.get_camera_state(cam_state_id)
            if cam_state is not None:
                camera_states.append(cam_state)
                observations.append(obs)

        if len(camera_states) < 3:
            return None, None

        # 多视角三角化
        point_world = Triangulation.triangulate_multi_view(camera_states, observations)
        if point_world is None:
            return None, None

        # 初步质量检查：整体重投影误差与正深度
        if not Triangulation.check_triangulation_quality(
            point_world, camera_states, observations, max_reprojection_error=3.0
        ):
            return None, None

        # 为有效视角逐个计算相机雅可比与残差，过滤负深度
        H_f_blocks = []   # 每视角的 2x3 特征雅可比
        H_x_blocks = []   # 每视角的 2x6 相机雅可比
        residual_blocks = []  # 每视角的 2x1 残差
        valid_views = []

        for cam_state, obs in zip(camera_states, observations):
            R_cam = cam_state.rotation_matrix
            p_cam_world = cam_state.position
            vec_world = point_world - p_cam_world
            point_cam = R_cam @ vec_world

            # 质量控制：仅保留正深度视角
            if point_cam[2] <= 0:
                continue

            # 投影雅可比（像素/归一化坐标对相机坐标的导数）
            H_proj = self.camera_model.compute_jacobian(point_cam)  # 2x3

            # 特征坐标雅可比（对世界点坐标）
            H_f = H_proj @ R_cam  # 2x3

            # 相机位姿误差雅可比（旋转小角与平移）
            H_x = np.zeros((2, 6))
            H_x[:, 0:3] = H_proj @ R_cam @ self._skew_symmetric(vec_world)
            H_x[:, 3:6] = -H_proj @ R_cam

            # 残差（观测 - 预测）
            u_proj = point_cam[0] / point_cam[2]
            v_proj = point_cam[1] / point_cam[2]
            residual = obs - np.array([u_proj, v_proj])

            H_f_blocks.append(H_f)
            H_x_blocks.append(H_x)
            residual_blocks.append(residual)
            valid_views.append(cam_state)

        S = len(H_f_blocks)
        if S < 3:
            # 有效视角不足，跳过更新
            return None, None

        # 堆叠所有视角的特征雅可比与残差：H_f_stack ∈ R^{2S×3}, r_stack ∈ R^{2S}
        H_f_stack = np.vstack(H_f_blocks)
        r_stack = np.concatenate(residual_blocks)

        # 计算左零空间用于一次性消元特征坐标
        L = self._compute_left_nullspace(H_f_stack)  # 2S×K
        K = L.shape[1]
        if K == 0:
            return None, None

        # 将每视角的相机雅可比堆叠到局部块矩阵 (2S × 6S)
        H_x_stack = np.zeros((2 * S, 6 * S))
        row = 0
        for i in range(S):
            H_x_stack[row:row + 2, i * 6:(i + 1) * 6] = H_x_blocks[i]
            row += 2

        # 投影到零空间：得到更强的约束
        H_proj_local = L.T @ H_x_stack      # (K × 6S)
        r_proj = L.T @ r_stack              # (K,)

        # 将局部块映射到全局状态雅可比 H_full
        H_full = np.zeros((K, self.state.state_dim))
        for i, cam_state in enumerate(valid_views):
            cam_idx = self._get_camera_state_index(cam_state.state_id)
            if cam_idx is None:
                continue
            H_full[:, cam_idx:cam_idx + 6] += H_proj_local[:, i * 6:(i + 1) * 6]

        return H_full, r_proj

    def _get_camera_state_index(self, state_id: int) -> Optional[int]:
        """Get index of camera state in full state vector."""
        sorted_ids = sorted(self.state.camera_states.keys())

        try:
            idx = sorted_ids.index(state_id)
            return 15 + idx * 6  # 15 for IMU state, 6 per camera
        except ValueError:
            return None

    def _apply_state_correction(self, correction: np.ndarray):
        """
        Apply error state correction to nominal state.

        Args:
            correction: Error state correction vector
        """
        # IMU state correction
        delta_theta = correction[0:3]
        delta_p = correction[3:6]
        delta_v = correction[6:9]
        delta_bg = correction[9:12]
        delta_ba = correction[12:15]

        # Orientation correction (small angle approximation)
        delta_q = self._small_angle_to_quaternion(delta_theta)
        self.state.imu_state.orientation = self._quaternion_multiply(
            self.state.imu_state.orientation,
            delta_q
        )
        # Normalize quaternion to remain on manifold
        self.state.imu_state.orientation = self._normalize_quaternion(self.state.imu_state.orientation)

        # Other state corrections
        self.state.imu_state.position += delta_p
        self.state.imu_state.velocity += delta_v
        self.state.imu_state.gyro_bias += delta_bg
        self.state.imu_state.accel_bias += delta_ba

        # Camera state corrections
        idx = 15
        # Collect per-camera orientation deltas for reset Jacobian
        cam_delta_thetas = {}
        for state_id in sorted(self.state.camera_states.keys()):
            cam_state = self.state.camera_states[state_id]

            delta_theta_cam = correction[idx:idx+3]
            delta_p_cam = correction[idx+3:idx+6]

            delta_q_cam = self._small_angle_to_quaternion(delta_theta_cam)
            cam_state.orientation = self._quaternion_multiply(
                cam_state.orientation,
                delta_q_cam
            )
            # Normalize quaternion to remain on manifold
            cam_state.orientation = self._normalize_quaternion(cam_state.orientation)
            cam_state.position += delta_p_cam

            # Store for covariance reset
            cam_delta_thetas[state_id] = delta_theta_cam
            idx += 6

        # On-manifold covariance reset: apply right-reset Jacobian to orientation error blocks
        # G is identity except for orientation 3x3 blocks (IMU and each camera):
        # G_theta = I - 0.5 * [delta_theta]_x, consistent with right-multiplicative update
        G = np.eye(self.ekf.state_dim)
        G[0:3, 0:3] = self._right_reset_jacobian(delta_theta)

        for state_id in sorted(self.state.camera_states.keys()):
            cam_idx = self._get_camera_state_index(state_id)
            if cam_idx is None:
                continue
            G[cam_idx:cam_idx+3, cam_idx:cam_idx+3] = self._right_reset_jacobian(cam_delta_thetas.get(state_id, np.zeros(3)))

        # Update covariance with reset mapping
        self.ekf.P = G @ self.ekf.P @ G.T
        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)

    def marginalize_camera_state(self, state_id: int):
        """
        Marginalize out a camera state.

        Args:
            state_id: ID of camera state to marginalize
        """
        cam_idx = self._get_camera_state_index(state_id)

        if cam_idx is None:
            return

        # Create mask for states to keep
        indices_to_keep = np.ones(self.state.state_dim, dtype=bool)
        indices_to_keep[cam_idx:cam_idx+6] = False

        # Marginalize covariance
        self.ekf.P = self.ekf.P[np.ix_(indices_to_keep, indices_to_keep)]
        self.ekf.state_dim = self.ekf.P.shape[0]

        # Remove from state
        self.state.remove_camera_state(state_id)

    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def _small_angle_to_quaternion(theta: np.ndarray) -> Quaternion:
        """Convert small angle vector to quaternion."""
        angle = np.linalg.norm(theta)
        if angle < 1e-8:
            return Quaternion(1.0, 0.0, 0.0, 0.0)

        axis = theta / angle
        half_angle = angle / 2
        s = np.sin(half_angle)

        return Quaternion(
            w=np.cos(half_angle),
            x=axis[0] * s,
            y=axis[1] * s,
            z=axis[2] * s
        )

    @staticmethod
    def _quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
        w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z

        return Quaternion(
            w=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            x=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        )

    @staticmethod
    def _normalize_quaternion(q: Quaternion) -> Quaternion:
        """Normalize quaternion to unit length."""
        norm = np.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
        if norm < 1e-12:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        inv = 1.0 / norm
        return Quaternion(w=q.w*inv, x=q.x*inv, y=q.y*inv, z=q.z*inv)

    @staticmethod
    def _right_reset_jacobian(delta_theta: np.ndarray) -> np.ndarray:
        """Right-multiplicative reset Jacobian for SO(3) small-angle injection.

        For q_new = q_old ⊗ exp(delta_theta), the orientation error reset mapping is
        approximated by G = I - 0.5 [delta_theta]_x.
        """
        return np.eye(3) - 0.5 * MSCKF._skew_symmetric(delta_theta)

    @staticmethod
    def _compute_left_nullspace(A: np.ndarray) -> np.ndarray:
        """Compute left null space (orthogonal complement of row space) of A.

        For A ∈ R^{m×n} with SVD A = U Σ V^T, the left null space is spanned by
        the last m - rank columns of U. Returns a basis matrix L ∈ R^{m×(m-rank)}
        such that L^T A = 0.
        """
        U, _, _ = np.linalg.svd(A, full_matrices=True)
        rank = np.linalg.matrix_rank(A)
        m = A.shape[0]
        if rank >= m:
            # No left null space; return empty with correct shape (m x 0)
            return np.zeros((m, 0))
        return U[:, rank:]

    def _create_initial_covariance(self) -> np.ndarray:
        """Create initial covariance matrix."""
        P = np.eye(15)

        # Orientation uncertainty
        P[0:3, 0:3] *= 0.01

        # Position uncertainty
        P[3:6, 3:6] *= 1.0

        # Velocity uncertainty
        P[6:9, 6:9] *= 0.1

        # Gyro bias uncertainty
        P[9:12, 9:12] *= 0.01

        # Accel bias uncertainty
        P[12:15, 12:15] *= 0.01

        return P

    def get_state(self) -> MSCKFState:
        """Get current state estimate."""
        return self.state.clone()

    def get_covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self.ekf.get_covariance()
