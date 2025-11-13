"""
Geographic feature management.

Manages geographic landmarks and map features for global localization.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from ..core.types import GeoFeature, Feature
from ..core.frames import ECEFFrame, ENUFrame
from ..core.state import MSCKFState, CameraState


class GeoFeatureManager:
    """
    Manager for geographic features (GPS landmarks, map points, etc.).

    Maintains a database of known geographic features and associates them
    with visual observations.
    """

    def __init__(self, ref_latitude: float, ref_longitude: float, ref_altitude: float):
        """
        Initialize geographic feature manager.

        Args:
            ref_latitude: Reference latitude in degrees
            ref_longitude: Reference longitude in degrees
            ref_altitude: Reference altitude in meters
        """
        self.ref_lat = ref_latitude
        self.ref_lon = ref_longitude
        self.ref_alt = ref_altitude

        # ENU frame for local coordinates
        self.enu_frame = ENUFrame(ref_latitude, ref_longitude, ref_altitude)

        # Geographic features: feature_id -> GeoFeature
        self.geo_features: Dict[int, GeoFeature] = {}

        # Visual-to-geographic associations: visual_feature_id -> geo_feature_id
        self.associations: Dict[int, int] = {}

        # Feature counter
        self._feature_id_counter = 0

    def add_geo_feature(self,
                       latitude: float,
                       longitude: float,
                       altitude: float,
                       horizontal_accuracy: float = 5.0,
                       vertical_accuracy: float = 10.0) -> int:
        """
        Add a geographic feature.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Altitude in meters
            horizontal_accuracy: Horizontal accuracy in meters
            vertical_accuracy: Vertical accuracy in meters

        Returns:
            Feature ID
        """
        feature = GeoFeature(
            feature_id=self._feature_id_counter,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            horizontal_accuracy=horizontal_accuracy,
            vertical_accuracy=vertical_accuracy
        )

        self.geo_features[self._feature_id_counter] = feature
        self._feature_id_counter += 1

        return feature.feature_id

    def associate_visual_feature(self, visual_feature_id: int, geo_feature_id: int):
        """
        Associate a visual feature with a geographic feature.

        Args:
            visual_feature_id: ID of visual feature
            geo_feature_id: ID of geographic feature
        """
        if geo_feature_id in self.geo_features:
            self.associations[visual_feature_id] = geo_feature_id

    def get_geo_feature(self, feature_id: int) -> Optional[GeoFeature]:
        """
        Get geographic feature by ID.

        Args:
            feature_id: Feature ID

        Returns:
            GeoFeature or None
        """
        return self.geo_features.get(feature_id)

    def get_geo_feature_for_visual(self, visual_feature_id: int) -> Optional[GeoFeature]:
        """
        Get geographic feature associated with visual feature.

        Args:
            visual_feature_id: Visual feature ID

        Returns:
            GeoFeature or None
        """
        geo_id = self.associations.get(visual_feature_id)
        if geo_id is not None:
            return self.geo_features.get(geo_id)
        return None

    def get_nearby_features(self,
                          latitude: float,
                          longitude: float,
                          radius: float = 100.0) -> List[GeoFeature]:
        """
        Get geographic features near a location.

        Args:
            latitude: Query latitude in degrees
            longitude: Query longitude in degrees
            radius: Search radius in meters

        Returns:
            List of nearby features
        """
        query_ecef = ECEFFrame.lla_to_ecef(latitude, longitude, 0)
        nearby = []

        for feature in self.geo_features.values():
            distance = np.linalg.norm(feature.position_ecef - query_ecef)
            if distance <= radius:
                nearby.append(feature)

        return nearby

    def compute_geo_measurement_jacobian(self,
                                        geo_feature: GeoFeature,
                                        camera_state: CameraState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute measurement Jacobian for geographic feature observation.

        Args:
            geo_feature: Geographic feature
            camera_state: Camera state

        Returns:
            (H, residual) Measurement Jacobian and residual
        """
        # Convert geographic feature to ENU
        feature_enu = self.enu_frame.ecef_to_enu(geo_feature.position_ecef)

        # Transform to camera frame
        R_cam = camera_state.rotation_matrix
        p_cam = camera_state.position

        point_cam = R_cam @ (feature_enu - p_cam)

        if point_cam[2] <= 0:
            return None, None

        # Projection Jacobian (2x3)
        Z = point_cam[2]
        H_proj = np.array([
            [1/Z, 0, -point_cam[0]/(Z*Z)],
            [0, 1/Z, -point_cam[1]/(Z*Z)]
        ])

        # Camera pose Jacobian (2x6)
        H_cam = np.zeros((2, 6))
        H_cam[:, 0:3] = H_proj @ R_cam @ self._skew_symmetric(feature_enu - p_cam)
        H_cam[:, 3:6] = -H_proj @ R_cam

        return H_cam, None  # Residual computed separately

    def to_enu(self, geo_feature: GeoFeature) -> np.ndarray:
        """
        Convert geographic feature to ENU coordinates.

        Args:
            geo_feature: Geographic feature

        Returns:
            ENU coordinates [east, north, up]
        """
        return self.enu_frame.ecef_to_enu(geo_feature.position_ecef)

    def from_enu(self, enu: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert ENU coordinates to LLA.

        Args:
            enu: ENU coordinates [east, north, up]

        Returns:
            (latitude, longitude, altitude) in degrees, degrees, meters
        """
        ecef = self.enu_frame.enu_to_ecef(enu)
        return ECEFFrame.ecef_to_lla(ecef)

    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])


class GeoConstraint:
    """
    Geographic constraint for global localization.

    Provides constraints from GPS measurements or known landmarks.
    """

    def __init__(self,
                 feature_manager: GeoFeatureManager,
                 position_noise: float = 5.0,
                 altitude_noise: float = 10.0):
        """
        Initialize geographic constraint.

        Args:
            feature_manager: Geographic feature manager
            position_noise: Horizontal position noise (meters)
            altitude_noise: Vertical position noise (meters)
        """
        self.feature_manager = feature_manager
        self.position_noise = position_noise
        self.altitude_noise = altitude_noise

    def compute_position_constraint(self,
                                   state: MSCKFState,
                                   measured_lat: float,
                                   measured_lon: float,
                                   measured_alt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute constraint from GPS position measurement.

        Args:
            state: Current MSCKF state
            measured_lat: Measured latitude (degrees)
            measured_lon: Measured longitude (degrees)
            measured_alt: Measured altitude (meters)

        Returns:
            (H, R, residual) Measurement model
        """
        # Convert measured position to ENU
        measured_enu = self.feature_manager.enu_frame.lla_to_enu(
            measured_lat, measured_lon, measured_alt
        )

        # Current position estimate in ENU
        estimated_enu = state.imu_state.position

        # Residual
        residual = measured_enu - estimated_enu

        # Jacobian (3x15)
        H = np.zeros((3, state.state_dim))
        H[:, 3:6] = np.eye(3)  # Position rows

        # Measurement noise
        R = np.diag([
            self.position_noise**2,
            self.position_noise**2,
            self.altitude_noise**2
        ])

        return H, R, residual

    def compute_velocity_constraint(self,
                                   state: MSCKFState,
                                   measured_velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute constraint from GPS velocity measurement.

        Args:
            state: Current MSCKF state
            measured_velocity: Measured velocity in ENU frame (3,)

        Returns:
            (H, R, residual) Measurement model
        """
        # Residual
        residual = measured_velocity - state.imu_state.velocity

        # Jacobian (3x15)
        H = np.zeros((3, state.state_dim))
        H[:, 6:9] = np.eye(3)  # Velocity rows

        # Measurement noise (assume 0.1 m/s)
        R = 0.1**2 * np.eye(3)

        return H, R, residual


class MapDatabase:
    """
    Database of known map features for global localization.

    Stores pre-built map with visual and geographic features.
    """

    def __init__(self, ref_latitude: float, ref_longitude: float, ref_altitude: float):
        """
        Initialize map database.

        Args:
            ref_latitude: Reference latitude
            ref_longitude: Reference longitude
            ref_altitude: Reference altitude
        """
        self.feature_manager = GeoFeatureManager(ref_latitude, ref_longitude, ref_altitude)

        # Map features: visual descriptors with geographic positions
        self.map_features: Dict[int, Dict] = {}

    def add_map_feature(self,
                       latitude: float,
                       longitude: float,
                       altitude: float,
                       descriptor: Optional[np.ndarray] = None):
        """
        Add feature to map database.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Altitude in meters
            descriptor: Visual descriptor (optional)
        """
        feature_id = self.feature_manager.add_geo_feature(
            latitude, longitude, altitude
        )

        self.map_features[feature_id] = {
            'geo_feature': self.feature_manager.get_geo_feature(feature_id),
            'descriptor': descriptor
        }

    def query_features(self,
                      camera_state: CameraState,
                      radius: float = 50.0) -> List[GeoFeature]:
        """
        Query features visible from camera location.

        Args:
            camera_state: Current camera state
            radius: Search radius in meters

        Returns:
            List of potentially visible features
        """
        # Convert camera position to LLA
        camera_enu = camera_state.position
        lat, lon, alt = self.feature_manager.from_enu(camera_enu)

        # Get nearby features
        return self.feature_manager.get_nearby_features(lat, lon, radius)

    def match_visual_to_map(self,
                           visual_features: List[Feature],
                           descriptors: np.ndarray,
                           camera_state: CameraState) -> List[Tuple[int, int]]:
        """
        Match visual features to map features.

        Args:
            visual_features: List of visual features
            descriptors: Visual descriptors (N, D)
            camera_state: Camera state

        Returns:
            List of matches (visual_id, map_id)
        """
        # Get potentially visible map features
        visible_map_features = self.query_features(camera_state)

        matches = []

        # Simple descriptor matching (can be improved)
        for i, (feature, desc) in enumerate(zip(visual_features, descriptors)):
            best_match = None
            best_distance = float('inf')

            for map_feature in visible_map_features:
                map_desc = self.map_features[map_feature.feature_id]['descriptor']

                if map_desc is not None:
                    distance = np.linalg.norm(desc - map_desc)

                    if distance < best_distance and distance < 0.7:  # Threshold
                        best_distance = distance
                        best_match = map_feature.feature_id

            if best_match is not None:
                matches.append((feature.feature_id, best_match))

        return matches

    def save(self, filename: str):
        """Save map database to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'features': self.map_features,
                'ref_lat': self.feature_manager.ref_lat,
                'ref_lon': self.feature_manager.ref_lon,
                'ref_alt': self.feature_manager.ref_alt
            }, f)

    @staticmethod
    def load(filename: str) -> 'MapDatabase':
        """Load map database from file."""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        db = MapDatabase(
            data['ref_lat'],
            data['ref_lon'],
            data['ref_alt']
        )
        db.map_features = data['features']

        return db
