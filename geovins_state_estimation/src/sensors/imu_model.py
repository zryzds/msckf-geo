"""
IMU sensor model for state propagation.

Implements continuous-time IMU dynamics and discrete-time integration.
"""
import numpy as np
from typing import List, Tuple
from ..core.types import IMUMeasurement, Quaternion
from ..core.state import IMUState


class IMUModel:
    """
    IMU sensor model with noise characteristics.

    The IMU provides angular velocity and linear acceleration measurements
    with additive white Gaussian noise and slowly varying bias.
    """

    def __init__(self,
                 gyro_noise_density: float = 1.6968e-04,  # rad/s/sqrt(Hz)
                 gyro_random_walk: float = 1.9393e-05,    # rad/s^2/sqrt(Hz)
                 accel_noise_density: float = 2.0000e-3,  # m/s^2/sqrt(Hz)
                 accel_random_walk: float = 3.0000e-3,    # m/s^3/sqrt(Hz)
                 gravity_magnitude: float = 9.81):
        """
        Initialize IMU model with noise parameters.

        Args:
            gyro_noise_density: Gyroscope white noise density
            gyro_random_walk: Gyroscope bias random walk
            accel_noise_density: Accelerometer white noise density
            accel_random_walk: Accelerometer bias random walk
            gravity_magnitude: Gravity magnitude in m/s^2
        """
        self.gyro_noise_density = gyro_noise_density
        self.gyro_random_walk = gyro_random_walk
        self.accel_noise_density = accel_noise_density
        self.accel_random_walk = accel_random_walk
        self.gravity = np.array([0.0, 0.0, -gravity_magnitude])

        # Process noise covariance (continuous time)
        self.Q_imu = np.diag([
            gyro_noise_density**2, gyro_noise_density**2, gyro_noise_density**2,
            gyro_random_walk**2, gyro_random_walk**2, gyro_random_walk**2,
            accel_noise_density**2, accel_noise_density**2, accel_noise_density**2,
            accel_random_walk**2, accel_random_walk**2, accel_random_walk**2
        ])

    def propagate(self,
                  state: IMUState,
                  measurements: List[IMUMeasurement]) -> IMUState:
        """
        Propagate IMU state using a batch of IMU measurements.

        Args:
            state: Current IMU state
            measurements: List of IMU measurements

        Returns:
            Propagated IMU state
        """
        current_state = state.clone()

        for i in range(len(measurements) - 1):
            dt = measurements[i + 1].timestamp - measurements[i].timestamp
            if dt <= 0:
                continue

            # Use midpoint integration
            omega_1 = measurements[i].angular_velocity - current_state.gyro_bias
            accel_1 = measurements[i].linear_acceleration - current_state.accel_bias

            # Simple Euler integration (can be upgraded to RK4)
            current_state = self._propagate_step(current_state, omega_1, accel_1, dt)
            current_state.timestamp = measurements[i + 1].timestamp

        return current_state

    def _propagate_step(self,
                       state: IMUState,
                       angular_vel: np.ndarray,
                       linear_accel: np.ndarray,
                       dt: float) -> IMUState:
        """
        Propagate state by one time step using IMU measurements.

        Args:
            state: Current state
            angular_vel: Angular velocity (bias corrected)
            linear_accel: Linear acceleration (bias corrected)
            dt: Time step

        Returns:
            Propagated state
        """
        # Current rotation matrix
        R = state.rotation_matrix

        # Propagate orientation using quaternion integration
        # dq/dt = 0.5 * Omega(omega) * q
        omega_norm = np.linalg.norm(angular_vel)
        if omega_norm > 1e-8:
            # Rodrigues formula for quaternion update
            angle = omega_norm * dt
            axis = angular_vel / omega_norm
            dq = self._axis_angle_to_quaternion(axis, angle)
            new_orientation = self._quaternion_multiply(state.orientation, dq)
        else:
            new_orientation = state.orientation

        # Propagate velocity: v_k+1 = v_k + (R * a - g) * dt
        accel_world = R @ linear_accel + self.gravity
        new_velocity = state.velocity + accel_world * dt

        # Propagate position: p_k+1 = p_k + v_k * dt + 0.5 * a * dt^2
        new_position = state.position + state.velocity * dt + 0.5 * accel_world * dt**2

        # Bias remains constant (random walk)
        new_gyro_bias = state.gyro_bias.copy()
        new_accel_bias = state.accel_bias.copy()

        return IMUState(
            orientation=new_orientation,
            position=new_position,
            velocity=new_velocity,
            gyro_bias=new_gyro_bias,
            accel_bias=new_accel_bias,
            state_id=state.state_id,
            timestamp=state.timestamp
        )

    def compute_state_transition(self,
                                 state: IMUState,
                                 angular_vel: np.ndarray,
                                 linear_accel: np.ndarray,
                                 dt: float) -> np.ndarray:
        """
        Compute discrete-time state transition matrix F.

        Args:
            state: Current state
            angular_vel: Angular velocity measurement (bias corrected)
            linear_accel: Linear acceleration measurement (bias corrected)
            dt: Time step

        Returns:
            State transition matrix F (15x15)
        """
        R = state.rotation_matrix

        # Build continuous-time system matrix
        F = np.zeros((15, 15))

        # Orientation error dynamics: d(delta_theta)/dt = -[omega]_x * delta_theta - n_g
        F[0:3, 0:3] = -self._skew_symmetric(angular_vel)
        F[0:3, 9:12] = -np.eye(3)

        # Position dynamics: dp/dt = v
        F[3:6, 6:9] = np.eye(3)

        # Velocity dynamics: dv/dt = -R * [a]_x * delta_theta - R * n_a
        F[6:9, 0:3] = -R @ self._skew_symmetric(linear_accel)
        F[6:9, 12:15] = -R

        # Bias dynamics: db/dt = 0 (random walk, driven by noise)

        # Discretize using matrix exponential approximation (first-order)
        Phi = np.eye(15) + F * dt

        return Phi

    def compute_process_noise(self, dt: float) -> np.ndarray:
        """
        Compute discrete-time process noise covariance.

        Args:
            dt: Time step

        Returns:
            Process noise covariance Q_d (15x15)
        """
        # Simplified discrete-time noise covariance
        Q_d = np.zeros((15, 15))

        # Orientation noise
        Q_d[0:3, 0:3] = self.gyro_noise_density**2 * dt**2 * np.eye(3)

        # Position noise (integrated from velocity)
        Q_d[3:6, 3:6] = (self.accel_noise_density**2 * dt**4 / 4) * np.eye(3)

        # Velocity noise
        Q_d[6:9, 6:9] = self.accel_noise_density**2 * dt**2 * np.eye(3)

        # Cross terms (position-velocity)
        Q_d[3:6, 6:9] = (self.accel_noise_density**2 * dt**3 / 2) * np.eye(3)
        Q_d[6:9, 3:6] = Q_d[3:6, 6:9].T

        # Gyro bias random walk
        Q_d[9:12, 9:12] = self.gyro_random_walk**2 * dt * np.eye(3)

        # Accel bias random walk
        Q_d[12:15, 12:15] = self.accel_random_walk**2 * dt * np.eye(3)

        return Q_d

    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def _axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> Quaternion:
        """Convert axis-angle to quaternion."""
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


class IMUIntegrator:
    """Higher-order IMU integration methods."""

    @staticmethod
    def rk4_step(state: IMUState,
                 omega: np.ndarray,
                 accel: np.ndarray,
                 dt: float,
                 gravity: np.ndarray) -> IMUState:
        """
        4th-order Runge-Kutta integration for IMU propagation.

        Args:
            state: Current state
            omega: Angular velocity (bias corrected)
            accel: Linear acceleration (bias corrected)
            dt: Time step
            gravity: Gravity vector

        Returns:
            Propagated state
        """
        # This is a simplified RK4; full implementation would query
        # IMU measurements at intermediate time steps

        # k1
        R1 = state.rotation_matrix
        a1 = R1 @ accel + gravity
        v1 = state.velocity
        p1 = state.position

        # k2 (midpoint)
        dt_half = dt / 2
        R2 = R1  # Simplified: should integrate rotation
        a2 = R2 @ accel + gravity
        v2 = v1 + a1 * dt_half
        p2 = p1 + v1 * dt_half

        # k3 (midpoint)
        R3 = R2
        a3 = R3 @ accel + gravity
        v3 = v1 + a2 * dt_half
        p3 = p1 + v2 * dt_half

        # k4 (endpoint)
        R4 = R3
        a4 = R4 @ accel + gravity
        v4 = v1 + a3 * dt
        p4 = p1 + v3 * dt

        # Combine
        new_velocity = v1 + (dt / 6) * (a1 + 2*a2 + 2*a3 + a4)
        new_position = p1 + (dt / 6) * (v1 + 2*v2 + 2*v3 + v4)

        # Orientation integration (simplified)
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-8:
            angle = omega_norm * dt
            axis = omega / omega_norm
            dq = IMUModel._axis_angle_to_quaternion(axis, angle)
            new_orientation = IMUModel._quaternion_multiply(state.orientation, dq)
        else:
            new_orientation = state.orientation

        return IMUState(
            orientation=new_orientation,
            position=new_position,
            velocity=new_velocity,
            gyro_bias=state.gyro_bias.copy(),
            accel_bias=state.accel_bias.copy(),
            state_id=state.state_id,
            timestamp=state.timestamp + dt
        )
