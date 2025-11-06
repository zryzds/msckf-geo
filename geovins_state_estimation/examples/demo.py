#!/usr/bin/env python3
"""
GEOVINS Demo: Geographic Visual-Inertial Navigation System

This demo shows how to use GEOVINS for state estimation with simulated data.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.state import create_initial_state
from src.core.types import IMUMeasurement, Feature, Quaternion, Pose
from src.sensors.imu_model import IMUModel
from src.sensors.camera_model import PinholeCameraModel
from src.filter.msckf import MSCKF
from src.features.tracker import FeatureTracker
from src.geographic.measurements import GPSMeasurement, GPSProcessor


class SimulatedDataGenerator:
    """Generate simulated sensor data for testing."""

    def __init__(self, duration=10.0, imu_rate=200.0, camera_rate=20.0, gps_rate=1.0):
        self.duration = duration
        self.imu_rate = imu_rate
        self.camera_rate = camera_rate
        self.gps_rate = gps_rate

        # Trajectory: circular motion
        self.radius = 10.0  # meters
        self.angular_velocity = 0.5  # rad/s

        # Reference location (San Francisco)
        self.ref_lat = 37.7749
        self.ref_lon = -122.4194
        self.ref_alt = 10.0

    def generate_imu_measurements(self):
        """Generate simulated IMU measurements."""
        dt = 1.0 / self.imu_rate
        num_samples = int(self.duration * self.imu_rate)

        measurements = []

        for i in range(num_samples):
            t = i * dt

            # Circular motion
            # angular_vel = [0, 0, omega]
            # centripetal acceleration = omega^2 * r
            omega = self.angular_velocity
            accel_centripetal = omega * omega * self.radius

            # Add gravity
            angular_velocity = np.array([0.0, 0.0, omega]) + np.random.randn(3) * 0.001
            linear_acceleration = np.array([accel_centripetal, 0.0, 9.81]) + np.random.randn(3) * 0.01

            measurements.append(IMUMeasurement(
                timestamp=t,
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration
            ))

        return measurements

    def generate_gps_measurements(self):
        """Generate simulated GPS measurements."""
        dt = 1.0 / self.gps_rate
        num_samples = int(self.duration * self.gps_rate)

        measurements = []

        for i in range(num_samples):
            t = i * dt

            # Circular motion in ENU
            angle = self.angular_velocity * t
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            z = 0.0

            # Add noise
            noise = np.random.randn(3) * np.array([5.0, 5.0, 10.0])
            position_enu = np.array([x, y, z]) + noise

            # Convert to LLA (simplified - just add to reference)
            # In practice, use proper ENU to LLA conversion
            lat = self.ref_lat + position_enu[1] / 111320.0  # Approximate
            lon = self.ref_lon + position_enu[0] / (111320.0 * np.cos(np.deg2rad(self.ref_lat)))
            alt = self.ref_alt + position_enu[2]

            measurements.append(GPSMeasurement(
                timestamp=t,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                horizontal_accuracy=5.0,
                vertical_accuracy=10.0,
                num_satellites=8
            ))

        return measurements

    def generate_features(self, timestamp):
        """Generate simulated feature observations."""
        # Random features in image
        num_features = 50
        features = []

        for i in range(num_features):
            u = np.random.uniform(100, 540)
            v = np.random.uniform(100, 380)
            features.append((u, v))

        return features


def run_demo():
    """Run the GEOVINS demo."""
    print("=" * 70)
    print("GEOVINS Demo: Geographic Visual-Inertial Navigation System")
    print("=" * 70)
    print()

    # Configuration
    print("1. Initializing system...")
    print("-" * 70)

    # Generate simulated data
    print("   - Generating simulated sensor data...")
    data_gen = SimulatedDataGenerator(duration=10.0)
    imu_measurements = data_gen.generate_imu_measurements()
    gps_measurements = data_gen.generate_gps_measurements()

    print(f"   - Generated {len(imu_measurements)} IMU measurements")
    print(f"   - Generated {len(gps_measurements)} GPS measurements")

    # Initialize system components
    print("   - Initializing MSCKF...")

    initial_state = create_initial_state(
        position=np.array([data_gen.radius, 0.0, 0.0]),
        velocity=np.zeros(3),
        orientation=Quaternion.identity()
    )

    # Set geographic reference
    initial_state.set_geo_reference(
        data_gen.ref_lat,
        data_gen.ref_lon,
        data_gen.ref_alt
    )

    imu_model = IMUModel(
        gyro_noise_density=1.6968e-04,
        accel_noise_density=2.0000e-03
    )

    camera_model = PinholeCameraModel(
        width=640, height=480,
        fx=458.654, fy=457.296,
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
        T_cam_imu=T_cam_imu,
        max_camera_states=15
    )

    # Initialize GPS processor
    gps_processor = GPSProcessor(
        ref_latitude=data_gen.ref_lat,
        ref_longitude=data_gen.ref_lon,
        ref_altitude=data_gen.ref_alt
    )

    # Initialize feature tracker
    feature_tracker = FeatureTracker(
        max_track_length=15,
        min_track_length=3,
        max_features=100
    )

    print("   - System initialized successfully!")
    print()

    # Main processing loop
    print("2. Processing sensor data...")
    print("-" * 70)

    camera_frame_counter = 0
    last_camera_time = 0.0
    camera_dt = 1.0 / data_gen.camera_rate

    gps_counter = 0
    trajectory = []

    # Process in time order
    imu_batch = []
    imu_idx = 0

    for t in np.arange(0, data_gen.duration, 0.01):
        # Collect IMU measurements
        while imu_idx < len(imu_measurements) and imu_measurements[imu_idx].timestamp <= t:
            imu_batch.append(imu_measurements[imu_idx])
            imu_idx += 1

        # Process IMU batch
        if len(imu_batch) >= 10:
            msckf.propagate_imu(imu_batch)
            imu_batch = []

        # Camera update
        if t - last_camera_time >= camera_dt:
            # Augment state with new camera pose
            msckf.augment_state(timestamp=t)

            # Generate and track features (simplified)
            features = data_gen.generate_features(t)
            new_features = feature_tracker.add_features(features[:20], camera_frame_counter)

            camera_frame_counter += 1
            last_camera_time = t

            # Get features ready for update
            ready_features = feature_tracker.get_features_for_update()
            if len(ready_features) > 0:
                # In practice, would perform full MSCKF update
                # msckf.update_features(ready_features)
                pass

        # GPS update
        if gps_counter < len(gps_measurements) and gps_measurements[gps_counter].timestamp <= t:
            gps_meas = gps_measurements[gps_counter]

            # Check quality
            if gps_processor.check_measurement_quality(gps_meas):
                # Compute measurement model
                H, R, residual = gps_processor.compute_measurement_model(
                    msckf.state, gps_meas
                )

                # Apply GPS update
                state_correction, _ = msckf.ekf.update(H, R, residual)
                msckf._apply_state_correction(state_correction)

            gps_counter += 1

        # Save trajectory
        if int(t * 10) % 10 == 0:  # Every 1 second
            state = msckf.get_state()
            trajectory.append({
                'timestamp': t,
                'position': state.imu_state.position.copy(),
                'velocity': state.imu_state.velocity.copy(),
                'num_camera_states': state.num_camera_states
            })

    print(f"   - Processed {imu_idx} IMU measurements")
    print(f"   - Processed {camera_frame_counter} camera frames")
    print(f"   - Processed {gps_counter} GPS measurements")
    print()

    # Display results
    print("3. Results")
    print("-" * 70)

    final_state = msckf.get_state()
    final_cov = msckf.get_covariance()

    print(f"   Final State:")
    print(f"   - Position:    {final_state.imu_state.position}")
    print(f"   - Velocity:    {final_state.imu_state.velocity}")
    print(f"   - Gyro bias:   {final_state.imu_state.gyro_bias}")
    print(f"   - Accel bias:  {final_state.imu_state.accel_bias}")
    print(f"   - Camera states: {final_state.num_camera_states}")
    print()

    print(f"   Uncertainty (position std dev):")
    position_cov = final_cov[3:6, 3:6]
    position_std = np.sqrt(np.diag(position_cov))
    print(f"   - Position:    {position_std}")
    print()

    # Display trajectory
    print("   Trajectory (sampled):")
    print("   " + "-" * 66)
    print("   {:>8s}  {:>12s}  {:>12s}  {:>12s}  {:>8s}".format(
        "Time(s)", "X(m)", "Y(m)", "Z(m)", "Cam States"
    ))
    print("   " + "-" * 66)

    for point in trajectory[::2]:  # Every 2 seconds
        pos = point['position']
        print("   {:8.2f}  {:12.3f}  {:12.3f}  {:12.3f}  {:8d}".format(
            point['timestamp'],
            pos[0], pos[1], pos[2],
            point['num_camera_states']
        ))

    print("   " + "-" * 66)
    print()

    # Summary statistics
    print("4. Summary")
    print("-" * 70)

    # Compute trajectory statistics
    positions = np.array([p['position'] for p in trajectory])
    velocities = np.array([p['velocity'] for p in trajectory])

    print(f"   Position range:")
    print(f"   - X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
    print(f"   - Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
    print(f"   - Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
    print()

    print(f"   Average velocity: {np.mean(np.linalg.norm(velocities, axis=1)):.3f} m/s")
    print()

    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


def main():
    """Main entry point."""
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
