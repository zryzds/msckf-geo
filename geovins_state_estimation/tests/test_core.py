"""
Unit tests for core modules.
"""
import pytest
import numpy as np
from src.core.types import (
    Quaternion, Pose, IMUMeasurement, Feature, GeoFeature, GeoMeasurement
)
from src.core.frames import (
    ECEFFrame, ENUFrame, FrameTransforms, IMUFrame, CameraFrame
)
from src.core.state import (
    IMUState, CameraState, MSCKFState, create_initial_state
)


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

    def test_rotations_orthonormal(self):
        # Check rotation matrices are orthonormal with det=1
        for angle in [0.0, 0.3, -0.7]:
            Rx = FrameTransforms.rotate_vector(
                Quaternion.from_rotation_matrix(ECEFFrame.rotation_matrix_x(angle)), np.array([1.0, 0.0, 0.0])
            )
            Ry = FrameTransforms.rotate_vector(
                Quaternion.from_rotation_matrix(ECEFFrame.rotation_matrix_y(angle)), np.array([0.0, 1.0, 0.0])
            )
            Rz = FrameTransforms.rotate_vector(
                Quaternion.from_rotation_matrix(ECEFFrame.rotation_matrix_z(angle)), np.array([0.0, 0.0, 1.0])
            )
            # Norms should remain 1
            assert np.isclose(np.linalg.norm(Rx), 1.0, atol=1e-8)
            assert np.isclose(np.linalg.norm(Ry), 1.0, atol=1e-8)
            assert np.isclose(np.linalg.norm(Rz), 1.0, atol=1e-8)


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

    def test_roundtrip_enu_ecef(self):
        ref_lat, ref_lon, ref_alt = 48.0, 11.0, 500.0
        enu_frame = ENUFrame(ref_lat, ref_lon, ref_alt)
        # Random small ENU vector
        enu = np.array([10.0, -5.0, 2.0])
        ecef = enu_frame.enu_to_ecef(enu)
        enu_back = enu_frame.ecef_to_enu(ecef)
        assert np.allclose(enu, enu_back, atol=1e-9)


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

    def test_rotation_matrix(self):
        # 90 deg yaw
        q = FrameTransforms.euler_to_quaternion(0.0, 0.0, np.pi/2)
        state = IMUState(
            orientation=q,
            position=np.zeros(3),
            velocity=np.zeros(3),
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3)
        )
        R = state.rotation_matrix
        v = np.array([1.0, 0.0, 0.0])
        v_rot = R @ v
        assert np.allclose(v_rot, np.array([0.0, 1.0, 0.0]), atol=1e-6)


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

    def test_clone_and_vector(self):
        state = create_initial_state(position=np.array([1.0, 2.0, 3.0]))
        # Add two camera states
        for i in range(2):
            cam_state = CameraState(
                state_id=i,
                timestamp=i * 0.1,
                orientation=Quaternion.identity(),
                position=np.array([i*0.1, 0.0, 0.0])
            )
            state.add_camera_state(cam_state)

        state.set_geo_reference(37.0, -122.0, 10.0)
        cloned = state.clone()
        # Modify original; clone should not change
        state.imu_state.position += np.array([10.0, 0.0, 0.0])
        assert np.allclose(cloned.imu_state.position, np.array([1.0, 2.0, 3.0]))

        vec = state.to_vector()
        assert vec.shape[0] == state.state_dim


class TestGeoTypes:
    """Tests for geographic types."""

    def test_geo_feature_ecef(self):
        lat, lon, alt = 30.0, 120.0, 50.0
        gf = GeoFeature(feature_id=1, latitude=lat, longitude=lon, altitude=alt)
        ecef_ref = ECEFFrame.lla_to_ecef(lat, lon, alt)
        assert np.allclose(gf.position_ecef, ecef_ref, atol=1e-6)

    def test_geo_measurement_to_ecef(self):
        gm = GeoMeasurement(timestamp=1.0, latitude=45.0, longitude=7.0, altitude=100.0)
        ecef = gm.to_ecef()
        ecef_ref = ECEFFrame.lla_to_ecef(45.0, 7.0, 100.0)
        assert np.allclose(ecef, ecef_ref, atol=1e-6)


class TestFramesUtilities:
    """Tests for frame utilities and camera transforms."""

    def test_quaternion_multiply_and_rotate(self):
        qx = FrameTransforms.euler_to_quaternion(np.pi/2, 0.0, 0.0)
        qy = FrameTransforms.euler_to_quaternion(0.0, np.pi/2, 0.0)
        qxy = FrameTransforms.quaternion_multiply(qx, qy)
        v = np.array([1.0, 1.0, 1.0])
        v_rot = FrameTransforms.rotate_vector(qxy, v)
        R = qxy.to_rotation_matrix()
        assert np.allclose(v_rot, R @ v, atol=1e-9)

    def test_imu_gravity_vector(self):
        g = IMUFrame.gravity_vector(9.81)
        assert np.allclose(g, np.array([0.0, 0.0, -9.81]))

    def test_camera_transform_roundtrip(self):
        # 90 deg yaw, 0.1m translation
        q = FrameTransforms.euler_to_quaternion(0.0, 0.0, np.pi/2)
        T = Pose(position=np.array([0.1, 0.0, 0.0]), orientation=q)
        cam_frame = CameraFrame(T_cam_imu=T)
        p_imu = np.array([0.2, -0.3, 0.5])
        p_cam = cam_frame.imu_to_camera(p_imu)
        p_imu_back = cam_frame.camera_to_imu(p_cam)
        assert np.allclose(p_imu, p_imu_back, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
