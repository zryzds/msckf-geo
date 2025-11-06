"""
Extended Kalman Filter (EKF) implementation.

Provides base EKF functionality for state estimation.
"""
import numpy as np
from typing import Tuple, Optional, Callable


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state estimation.

    Implements prediction and update steps with error-state formulation.
    """

    def __init__(self, state_dim: int, initial_covariance: Optional[np.ndarray] = None):
        """
        Initialize EKF.

        Args:
            state_dim: Dimension of state vector
            initial_covariance: Initial state covariance (default: identity)
        """
        self.state_dim = state_dim

        if initial_covariance is None:
            self.P = np.eye(state_dim)
        else:
            self.P = initial_covariance.copy()

        # Small value for numerical stability
        self.epsilon = 1e-10

    def predict(self, F: np.ndarray, Q: np.ndarray):
        """
        EKF prediction step.

        P_k+1 = F * P_k * F^T + Q

        Args:
            F: State transition matrix
            Q: Process noise covariance
        """
        self.P = F @ self.P @ F.T + Q

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # Ensure positive definiteness
        self._ensure_positive_definite()

    def update(self,
               H: np.ndarray,
               R: np.ndarray,
               residual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF update step.

        Args:
            H: Measurement Jacobian
            R: Measurement noise covariance
            residual: Innovation (measurement residual)

        Returns:
            (state_correction, updated_covariance)
        """
        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + R

        # Ensure symmetry
        S = 0.5 * (S + S.T)

        # Kalman gain: K = P * H^T * S^-1
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            K = self.P @ H.T @ np.linalg.pinv(S)

        # State correction: dx = K * residual
        state_correction = K @ residual

        # Covariance update: P = (I - K*H) * P * (I - K*H)^T + K*R*K^T (Joseph form)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # Ensure positive definiteness
        self._ensure_positive_definite()

        return state_correction, self.P

    def marginalize(self, indices_to_keep: np.ndarray):
        """
        Marginalize out states by keeping only specified indices.

        Args:
            indices_to_keep: Boolean array or indices of states to keep
        """
        self.P = self.P[np.ix_(indices_to_keep, indices_to_keep)]
        self.state_dim = self.P.shape[0]

    def augment(self, new_state_cov: np.ndarray, cross_cov: Optional[np.ndarray] = None):
        """
        Augment state with new variables.

        Args:
            new_state_cov: Covariance of new state
            cross_cov: Cross-covariance between existing and new state
        """
        old_dim = self.state_dim
        new_dim = new_state_cov.shape[0]
        total_dim = old_dim + new_dim

        # Create augmented covariance
        P_aug = np.zeros((total_dim, total_dim))

        # Copy existing covariance
        P_aug[:old_dim, :old_dim] = self.P

        # New state covariance
        P_aug[old_dim:, old_dim:] = new_state_cov

        # Cross-covariance
        if cross_cov is not None:
            P_aug[:old_dim, old_dim:] = cross_cov
            P_aug[old_dim:, :old_dim] = cross_cov.T

        self.P = P_aug
        self.state_dim = total_dim

    def get_covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self.P.copy()

    def set_covariance(self, P: np.ndarray):
        """Set state covariance."""
        self.P = P.copy()
        self.state_dim = P.shape[0]

    def _ensure_positive_definite(self):
        """Ensure covariance matrix is positive definite."""
        # Add small diagonal term if needed
        min_eigenvalue = np.min(np.linalg.eigvalsh(self.P))

        if min_eigenvalue < self.epsilon:
            self.P += (self.epsilon - min_eigenvalue) * np.eye(self.state_dim)


class InformationFilter:
    """
    Information form of the Kalman filter.

    Maintains information matrix (inverse covariance) instead of covariance.
    Can be more efficient for certain operations.
    """

    def __init__(self, state_dim: int, initial_information: Optional[np.ndarray] = None):
        """
        Initialize information filter.

        Args:
            state_dim: Dimension of state vector
            initial_information: Initial information matrix (default: identity)
        """
        self.state_dim = state_dim

        if initial_information is None:
            self.Y = np.eye(state_dim)  # Information matrix
        else:
            self.Y = initial_information.copy()

    def predict(self, F: np.ndarray, Q: np.ndarray):
        """
        Information filter prediction.

        Args:
            F: State transition matrix
            Q: Process noise covariance
        """
        # Convert to covariance form for prediction
        P = np.linalg.inv(self.Y)
        P_pred = F @ P @ F.T + Q

        # Convert back to information form
        self.Y = np.linalg.inv(P_pred)

    def update(self, H: np.ndarray, R: np.ndarray, measurement: np.ndarray):
        """
        Information filter update.

        Args:
            H: Measurement Jacobian
            R: Measurement noise covariance
            measurement: Measurement vector
        """
        # Information update
        R_inv = np.linalg.inv(R)
        self.Y += H.T @ R_inv @ H

    def get_covariance(self) -> np.ndarray:
        """Get state covariance (inverse of information matrix)."""
        return np.linalg.inv(self.Y)


class SquareRootFilter:
    """
    Square-root form of EKF for improved numerical stability.

    Maintains Cholesky factor of covariance instead of covariance itself.
    """

    def __init__(self, state_dim: int, initial_covariance: Optional[np.ndarray] = None):
        """
        Initialize square-root filter.

        Args:
            state_dim: Dimension of state vector
            initial_covariance: Initial state covariance
        """
        self.state_dim = state_dim

        if initial_covariance is None:
            self.S = np.eye(state_dim)  # Cholesky factor
        else:
            self.S = np.linalg.cholesky(initial_covariance)

    def predict(self, F: np.ndarray, Q: np.ndarray):
        """
        Square-root prediction using QR decomposition.

        Args:
            F: State transition matrix
            Q: Process noise covariance
        """
        # Compute square root of Q
        Q_sqrt = np.linalg.cholesky(Q)

        # Form matrix for QR decomposition
        # [F*S  Q_sqrt]
        A = np.column_stack([F @ self.S, Q_sqrt])

        # QR decomposition
        _, R = np.linalg.qr(A.T)

        # Extract square root of predicted covariance
        self.S = R[:self.state_dim, :].T

    def update(self, H: np.ndarray, R: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """
        Square-root update using QR decomposition.

        Args:
            H: Measurement Jacobian
            R: Measurement noise covariance
            residual: Measurement residual

        Returns:
            State correction
        """
        # Compute square root of R
        R_sqrt = np.linalg.cholesky(R)

        # Form matrix for QR decomposition
        # [R_sqrt         0    ]
        # [H*S      S^T*H^T*S  ]
        measurement_dim = H.shape[0]

        A = np.vstack([
            np.column_stack([R_sqrt, np.zeros((measurement_dim, self.state_dim))]),
            np.column_stack([H @ self.S, self.S.T])
        ])

        # QR decomposition
        _, R_qr = np.linalg.qr(A.T)

        # Extract updated square root
        self.S = R_qr[measurement_dim:, measurement_dim:].T

        # Compute Kalman gain and state correction
        P = self.S @ self.S.T
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state_correction = K @ residual

        return state_correction

    def get_covariance(self) -> np.ndarray:
        """Get state covariance."""
        return self.S @ self.S.T


class AdaptiveEKF(ExtendedKalmanFilter):
    """
    Adaptive EKF that adjusts process/measurement noise online.

    Uses innovation sequence to adapt noise parameters.
    """

    def __init__(self,
                 state_dim: int,
                 initial_covariance: Optional[np.ndarray] = None,
                 adaptation_rate: float = 0.95):
        """
        Initialize adaptive EKF.

        Args:
            state_dim: Dimension of state vector
            initial_covariance: Initial state covariance
            adaptation_rate: Exponential smoothing rate for innovation
        """
        super().__init__(state_dim, initial_covariance)

        self.adaptation_rate = adaptation_rate
        self.innovation_history = []
        self.max_history = 50

    def update(self,
               H: np.ndarray,
               R: np.ndarray,
               residual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive update with noise estimation.

        Args:
            H: Measurement Jacobian
            R: Measurement noise covariance
            residual: Innovation

        Returns:
            (state_correction, updated_covariance)
        """
        # Store innovation
        self.innovation_history.append(residual)
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)

        # Perform standard update
        state_correction, P = super().update(H, R, residual)

        return state_correction, P

    def estimate_measurement_noise(self, H: np.ndarray) -> np.ndarray:
        """
        Estimate measurement noise covariance from innovation sequence.

        Args:
            H: Measurement Jacobian

        Returns:
            Estimated measurement noise covariance
        """
        if len(self.innovation_history) < 2:
            return None

        # Compute innovation covariance
        innovations = np.array(self.innovation_history)
        C_v = np.cov(innovations.T)

        # Estimate R: C_v = H*P*H^T + R
        # R = C_v - H*P*H^T
        R_est = C_v - H @ self.P @ H.T

        return R_est
