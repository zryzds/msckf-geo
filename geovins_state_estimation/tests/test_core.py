"""
Unit tests for core modules.
"""
import pytest
import numpy as np
from src.core.types import Quaternion, Pose, IMUMeasurement, Feature
from src.core.frames import ECEFFrame, ENUFrame, FrameTransforms
from src.core.state import IMUState, CameraState, MSCKFState, create_initial_state


class TestQuaternion:
    """Tests for Quaternion class."""

    def test_identity(self):
        q = Quaternion.identity()
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_normalization(self):
        q = Quaternion(2.0, 0.0, 0.0, 0.0)
        assert np.isclose(q.w, 1.0)

    def test_to_rotation_matrix(self):
        q = Quaternion.identity()
        R = q.to_rotation_matrix()
        assert np.allclose(R, np.eye(3))

    def test_from_rotation_matrix(self):
        R = np.eye(3)
        q = Quaternion.from_rotation_matrix(R)
        assert np.isclose(q.w, 1.0)

    def test_euler_conversion(self):
        roll, pitch, yaw = 0.1, 0.2, 0.3
        q = FrameTransforms.euler_to_quaternion(roll, pitch, yaw)
        euler_back = FrameTransforms.quaternion_to_euler(q)
        assert np.allclose([roll, pitch, yaw], euler_back, atol=1e-6)


class TestPose:
    """Tests for Pose class."""

    def test_identity_pose(self):
        pose = Pose(
            position=np.zeros(3),
            orientation=Quaternion.identity()
        )
        T = pose.to_matrix()
        assert np.allclose(T, np.eye(4))

    def test_inverse(self):
        pose = Pose(
            position=np.array([1.0, 2.0, 3.0]),
            orientation=Quaternion.identity()
        )
        inv_pose = pose.inverse()

        # Apply pose then inverse should give identity
        T = pose.to_matrix()
        T_inv = inv_pose.to_matrix()
        assert np.allclose(T @ T_inv, np.eye(4), atol=1e-10)


class TestIMUMeasurement:
    """Tests for IMU measurement."""

    def test_creation(self):
        imu_meas = IMUMeasurement(
            timestamp=1.0,
            angular_velocity=[0.1, 0.2, 0.3],
            linear_acceleration=[0.0, 0.0, 9.81]
        )
        assert imu_meas.timestamp == 1.0
        assert isinstance(imu_meas.angular_velocity, np.ndarray)
        assert isinstance(imu_meas.linear_acceleration, np.ndarray)


class TestFeature:
    """Tests for Feature class."""

    def test_feature_creation(self):
        feature = Feature(
            feature_id=0,
            u=100.5,
            v=200.3
        )
        assert feature.feature_id == 0
        assert feature.num_observations == 0

    def test_add_observation(self):
        feature = Feature(feature_id=0, u=100.0, v=200.0)
        feature.add_observation(0, np.array([100.0, 200.0]))
        feature.add_observation(1, np.array([101.0, 201.0]))
        assert feature.num_observations == 2


class TestECEFFrame:
    """Tests for ECEF coordinate frame."""

    def test_lla_to_ecef_to_lla(self):
        # Test location: San Francisco
        lat, lon, alt = 37.7749, -122.4194, 10.0

        # Convert to ECEF
        ecef = ECEFFrame.lla_to_ecef(lat, lon, alt)

        # Convert back
        lat_back, lon_back, alt_back = ECEFFrame.ecef_to_lla(ecef)

        assert np.isclose(lat, lat_back, atol=1e-6)
        assert np.isclose(lon, lon_back, atol=1e-6)
        assert np.isclose(alt, alt_back, atol=1e-3)

    def test_equator(self):
        # Test at equator
        ecef = ECEFFrame.lla_to_ecef(0.0, 0.0, 0.0)
        assert ecef[2] < 1.0  # Should be near zero


class TestENUFrame:
    """Tests for ENU coordinate frame."""

    def test_ecef_to_enu_to_ecef(self):
        ref_lat, ref_lon, ref_alt = 37.0, -122.0, 0.0
        enu_frame = ENUFrame(ref_lat, ref_lon, ref_alt)

        # Point at reference should be origin in ENU
        ref_ecef = ECEFFrame.lla_to_ecef(ref_lat, ref_lon, ref_alt)
        enu = enu_frame.ecef_to_enu(ref_ecef)

        assert np.allclose(enu, np.zeros(3), atol=1e-3)

    def test_lla_to_enu(self):
        ref_lat, ref_lon, ref_alt = 37.0, -122.0, 0.0
        enu_frame = ENUFrame(ref_lat, ref_lon, ref_alt)

        # Point slightly north
        enu = enu_frame.lla_to_enu(37.001, -122.0, 0.0)

        # Should have positive north component
        assert enu[1] > 0


class TestIMUState:
    """Tests for IMU state."""

    def test_creation(self):
        state = IMUState(
            orientation=Quaternion.identity(),
            position=np.zeros(3),
            velocity=np.zeros(3),
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3)
        )
        assert isinstance(state.position, np.ndarray)
        assert state.position.shape == (3,)

    def test_clone(self):
        state = IMUState(
            orientation=Quaternion.identity(),
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.zeros(3),
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3)
        )
        cloned = state.clone()

        # Modify original
        state.position[0] = 999.0

        # Clone should be unchanged
        assert cloned.position[0] == 1.0


class TestMSCKFState:
    """Tests for MSCKF state."""

    def test_creation(self):
        state = create_initial_state()
        assert state.num_camera_states == 0
        assert state.state_dim == 15  # Just IMU state

    def test_add_camera_state(self):
        state = create_initial_state()

        cam_state = CameraState(
            state_id=0,
            timestamp=1.0,
            orientation=Quaternion.identity(),
            position=np.zeros(3)
        )

        state.add_camera_state(cam_state)

        assert state.num_camera_states == 1
        assert state.state_dim == 21  # 15 + 6

    def test_remove_camera_state(self):
        state = create_initial_state()

        cam_state = CameraState(
            state_id=0,
            timestamp=1.0,
            orientation=Quaternion.identity(),
            position=np.zeros(3)
        )

        state.add_camera_state(cam_state)
        state.remove_camera_state(0)

        assert state.num_camera_states == 0

    def test_max_camera_states(self):
        state = create_initial_state()
        state.max_camera_states = 5

        # Add more than max
        for i in range(10):
            cam_state = CameraState(
                state_id=i,
                timestamp=i * 0.1,
                orientation=Quaternion.identity(),
                position=np.zeros(3)
            )
            state.add_camera_state(cam_state)

        # Should only keep max
        assert state.num_camera_states == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
