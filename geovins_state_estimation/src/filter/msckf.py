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

    def update_features(self, features: List[Feature]):
        """
        Perform measurement update using features.

        Args:
            features: List of features ready for update
        """
        if len(features) == 0:
            return

        # Build measurement model for all features
        H_all = []
        r_all = []

        for feature in features:
            H_f, r_f = self._compute_feature_jacobian(feature)

            if H_f is not None and r_f is not None:
                H_all.append(H_f)
                r_all.append(r_f)

        if len(H_all) == 0:
            return

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

    def _compute_feature_jacobian(self, feature: Feature) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute measurement Jacobian for a feature.

        Args:
            feature: Feature with multiple observations

        Returns:
            (H, r) Jacobian and residual, or (None, None) if triangulation fails
        """
        if feature.num_observations < 2:
            return None, None

        # Get camera states and observations
        camera_states = []
        observations = []

        for cam_state_id, obs in feature.observations:
            cam_state = self.state.get_camera_state(cam_state_id)
            if cam_state is not None:
                camera_states.append(cam_state)
                observations.append(obs)

        if len(camera_states) < 2:
            return None, None

        # Triangulate feature
        point_world = Triangulation.triangulate_multi_view(camera_states, observations)

        if point_world is None:
            return None, None

        # Check triangulation quality
        if not Triangulation.check_triangulation_quality(
            point_world, camera_states, observations, max_reprojection_error=3.0
        ):
            return None, None

        # Compute Jacobians for each observation
        H_f_list = []
        r_f_list = []

        for cam_state, obs in zip(camera_states, observations):
            # Transform point to camera frame
            R_cam = cam_state.rotation_matrix
            p_cam_world = cam_state.position
            point_cam = R_cam @ (point_world - p_cam_world)

            if point_cam[2] <= 0:
                continue

            # Projection Jacobian
            H_proj = self.camera_model.compute_jacobian(point_cam)

            # Feature position Jacobian
            H_f = H_proj @ R_cam

            # Camera pose Jacobian
            H_x = np.zeros((2, 6))
            H_x[:, 0:3] = H_proj @ R_cam @ self._skew_symmetric(point_world - p_cam_world)
            H_x[:, 3:6] = -H_proj @ R_cam

            # Null space projection (eliminate feature depth)
            # Use left null space of H_f to eliminate feature
            A = self._compute_left_nullspace(H_f)
            H_x_proj = A.T @ H_x

            # Residual
            u_proj = point_cam[0] / point_cam[2]
            v_proj = point_cam[1] / point_cam[2]
            residual = obs - np.array([u_proj, v_proj])
            r_proj = A.T @ residual

            H_f_list.append(H_x_proj)
            r_f_list.append(r_proj)

        if len(H_f_list) == 0:
            return None, None

        # Stack Jacobians
        H_x = np.vstack(H_f_list)
        r_x = np.concatenate(r_f_list)

        # Build full Jacobian (map to state indices)
        H_full = np.zeros((H_x.shape[0], self.state.state_dim))

        row_idx = 0
        for i, (cam_state, _) in enumerate(zip(camera_states, observations)):
            # Find camera state index in full state
            cam_idx = self._get_camera_state_index(cam_state.state_id)

            if cam_idx is not None:
                block_size = H_f_list[i].shape[0]
                H_full[row_idx:row_idx+block_size, cam_idx:cam_idx+6] = H_f_list[i]
                row_idx += block_size

        return H_full[:row_idx, :], r_x[:row_idx]

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

        # Other state corrections
        self.state.imu_state.position += delta_p
        self.state.imu_state.velocity += delta_v
        self.state.imu_state.gyro_bias += delta_bg
        self.state.imu_state.accel_bias += delta_ba

        # Camera state corrections
        idx = 15
        for state_id in sorted(self.state.camera_states.keys()):
            cam_state = self.state.camera_states[state_id]

            delta_theta_cam = correction[idx:idx+3]
            delta_p_cam = correction[idx+3:idx+6]

            delta_q_cam = self._small_angle_to_quaternion(delta_theta_cam)
            cam_state.orientation = self._quaternion_multiply(
                cam_state.orientation,
                delta_q_cam
            )
            cam_state.position += delta_p_cam

            idx += 6

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
    def _compute_left_nullspace(A: np.ndarray) -> np.ndarray:
        """Compute left null space of matrix A."""
        _, _, Vt = np.linalg.svd(A)
        # Left null space is orthogonal complement of row space
        rank = np.linalg.matrix_rank(A)
        return Vt[rank:, :].T

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
