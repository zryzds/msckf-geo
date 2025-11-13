"""
Unit tests for filter modules.
"""
import pytest
import numpy as np
import os
import csv
import json
from src.filter.ekf import ExtendedKalmanFilter
from src.filter.msckf import MSCKF
from src.core.state import create_initial_state
from src.core.types import IMUMeasurement, Quaternion, Pose
from src.sensors.imu_model import IMUModel
from src.sensors.camera_model import PinholeCameraModel


class TestEKF:
    """Tests for Extended Kalman Filter."""

    def test_creation(self):
        ekf = ExtendedKalmanFilter(state_dim=15)
        assert ekf.state_dim == 15
        assert ekf.P.shape == (15, 15)

    def test_predict(self):
        ekf = ExtendedKalmanFilter(state_dim=3)
        P_init = ekf.P.copy()

        F = np.eye(3)
        Q = 0.1 * np.eye(3)

        ekf.predict(F, Q)

        # Covariance should increase
        assert np.any(ekf.P != P_init)

    def test_update(self):
        ekf = ExtendedKalmanFilter(state_dim=3)
        P_init = ekf.P.copy()

        H = np.eye(3)
        R = 0.1 * np.eye(3)
        residual = np.array([0.1, 0.2, 0.3])

        correction, P_updated = ekf.update(H, R, residual)

        # Covariance should decrease
        assert np.trace(P_updated) < np.trace(P_init)

    def test_marginalize(self):
        ekf = ExtendedKalmanFilter(state_dim=10)

        # Keep only first 5 dimensions
        indices = np.arange(5)
        ekf.marginalize(indices)

        assert ekf.state_dim == 5
        assert ekf.P.shape == (5, 5)

    def test_augment(self):
        ekf = ExtendedKalmanFilter(state_dim=5)

        new_state_cov = np.eye(3)
        ekf.augment(new_state_cov)

        assert ekf.state_dim == 8
        assert ekf.P.shape == (8, 8)


class TestMSCKF:
    """Tests for MSCKF."""

    @pytest.fixture
    def msckf_system(self):
        """Create MSCKF system for testing."""
        initial_state = create_initial_state()

        imu_model = IMUModel(
            gyro_noise_density=1e-4,
            accel_noise_density=2e-3
        )

        camera_model = PinholeCameraModel(
            width=640, height=480,
            fx=458.0, fy=458.0,
            cx=320.0, cy=240.0
        )

        T_cam_imu = Pose(
            position=np.zeros(3),
            orientation=Quaternion.identity()
        )

        msckf = MSCKF(
            initial_state=initial_state,
            imu_model=imu_model,
            camera_model=camera_model,
            T_cam_imu=T_cam_imu
        )

        return msckf

    def test_creation(self, msckf_system):
        assert msckf_system.state.num_camera_states == 0

    def test_imu_propagation(self, msckf_system):
        # Create IMU measurements (stationary)
        imu_measurements = [
            IMUMeasurement(
                timestamp=i * 0.005,
                angular_velocity=np.zeros(3),
                linear_acceleration=np.array([0.0, 0.0, 9.81])
            )
            for i in range(100)
        ]

        initial_pos = msckf_system.state.imu_state.position.copy()

        msckf_system.propagate_imu(imu_measurements)

        # For stationary system, position should remain close
        final_pos = msckf_system.state.imu_state.position
        assert np.linalg.norm(final_pos - initial_pos) < 0.1

    def test_state_augmentation(self, msckf_system):
        initial_dim = msckf_system.state.state_dim

        msckf_system.augment_state(timestamp=1.0)

        # State dimension should increase by 6
        assert msckf_system.state.state_dim == initial_dim + 6
        assert msckf_system.state.num_camera_states == 1

    def test_camera_state_marginalization(self, msckf_system):
        # Augment state
        msckf_system.augment_state(timestamp=1.0)
        camera_state_id = msckf_system.state.get_sorted_camera_states()[0].state_id

        initial_dim = msckf_system.state.state_dim

        # Marginalize
        msckf_system.marginalize_camera_state(camera_state_id)

        # State dimension should decrease by 6
        assert msckf_system.state.state_dim == initial_dim - 6
        assert msckf_system.state.num_camera_states == 0

    def test_get_state(self, msckf_system):
        state = msckf_system.get_state()
        assert state is not None
        assert hasattr(state, 'imu_state')

    def test_get_covariance(self, msckf_system):
        P = msckf_system.get_covariance()
        assert P.shape[0] == msckf_system.state.state_dim
        assert P.shape[1] == msckf_system.state.state_dim


class TestIMUPropagation:
    """Test IMU propagation accuracy."""

    def test_constant_velocity(self):
        """Test propagation with constant velocity."""
        initial_state = create_initial_state(
            velocity=np.array([1.0, 0.0, 0.0])  # 1 m/s in x direction
        )

        imu_model = IMUModel()

        # Create measurements (no rotation, compensate gravity)
        dt = 0.01
        num_steps = 100
        imu_measurements = [
            IMUMeasurement(
                timestamp=i * dt,
                angular_velocity=np.zeros(3),
                linear_acceleration=np.array([0.0, 0.0, 9.81])
            )
            for i in range(num_steps)
        ]

        final_state = imu_model.propagate(initial_state.imu_state, imu_measurements)

        # After 1 second, should move 1 meter in x
        expected_displacement = 1.0  # 1 m/s * 1 s
        actual_displacement = final_state.position[0]

        assert np.isclose(actual_displacement, expected_displacement, atol=0.1)


class TestIMUFromCSV:
    """Use real IMU CSV to validate MSCKF propagation sanity."""

    CSV_PATH = r"c:\Users\Admin\Desktop\msckf-geo\TD_07\export\imu.csv"

    @staticmethod
    def read_imu_csv(path: str, max_rows: int = 2000):
        measurements = []
        if not os.path.exists(path):
            return measurements

        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return measurements

            # Try to locate indices by header names
            def idx(name, default=None):
                try:
                    return header.index(name)
                except ValueError:
                    return default

            ts_i = idx('timestamp', 1)
            wx_i = idx('wx', 5)
            wy_i = idx('wy', 6)
            wz_i = idx('wz', 7)
            ax_i = idx('ax', 8)
            ay_i = idx('ay', 9)
            az_i = idx('az', 10)

            count = 0
            for row in reader:
                if max_rows is not None and count >= max_rows:
                    break
                try:
                    t = float(row[ts_i])
                    w = np.array([
                        float(row[wx_i]),
                        float(row[wy_i]),
                        float(row[wz_i])
                    ], dtype=np.float64)
                    a = np.array([
                        float(row[ax_i]),
                        float(row[ay_i]),
                        float(row[az_i])
                    ], dtype=np.float64)
                except (ValueError, IndexError):
                    continue

                measurements.append(IMUMeasurement(timestamp=t, angular_velocity=w, linear_acceleration=a))
                count += 1

        # Ensure strictly increasing timestamps
        measurements = [m for i, m in enumerate(measurements) if i == 0 or (measurements[i].timestamp > measurements[i-1].timestamp)]
        return measurements

    def test_msckf_with_csv(self):
        csv_path = self.CSV_PATH
        if not os.path.exists(csv_path):
            pytest.skip(f"IMU CSV not found: {csv_path}")

        # Load first N rows to keep test time reasonable
        imu_measurements = self.read_imu_csv(csv_path, max_rows=2000)
        assert len(imu_measurements) > 10

        # Build MSCKF system
        initial_state = create_initial_state(timestamp=imu_measurements[0].timestamp)

        imu_model = IMUModel()
        camera_model = PinholeCameraModel(
            width=640, height=480,
            fx=458.0, fy=458.0,
            cx=320.0, cy=240.0
        )
        T_cam_imu = Pose(position=np.zeros(3), orientation=Quaternion.identity())

        msckf = MSCKF(
            initial_state=initial_state,
            imu_model=imu_model,
            camera_model=camera_model,
            T_cam_imu=T_cam_imu
        )

        initial_pos = msckf.state.imu_state.position.copy()

        # Propagate with real IMU
        msckf.propagate_imu(imu_measurements)

        final_pos = msckf.state.imu_state.position
        displacement = float(np.linalg.norm(final_pos - initial_pos))

        # Covariance sanity: symmetric and positive semi-definite
        P = msckf.get_covariance()
        min_eig = float(np.min(np.linalg.eigvalsh(P)))

        # dt stats
        ts = np.array([m.timestamp for m in imu_measurements])
        dts = np.diff(ts)
        dt_median = float(np.median(dts)) if len(dts) else 0.0
        dt_min = float(np.min(dts)) if len(dts) else 0.0
        dt_max = float(np.max(dts)) if len(dts) else 0.0

        # Persist metrics for report
        metrics = {
            "csv_path": csv_path,
            "num_measurements": len(imu_measurements),
            "dt_median": dt_median,
            "dt_min": dt_min,
            "dt_max": dt_max,
            "final_position": msckf.state.imu_state.position.tolist(),
            "final_velocity": msckf.state.imu_state.velocity.tolist(),
            "displacement": displacement,
            "cov_min_eigenvalue": min_eig,
        }

        out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "output"))
        out_path = os.path.join(out_dir, "imu_csv_metrics.json")
        try:
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            # If path not writable in CI, ignore
            pass

        # Assertions: timestamps increasing, finite displacement, covariance PSD-ish
        assert dt_median > 0
        assert not np.isnan(displacement)
        assert min_eig > -1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
