"""
Feature tracking module.

Tracks visual features across camera frames.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from ..core.types import Feature, FeatureStatus
from collections import defaultdict


class FeatureTracker:
    """
    Feature tracker for maintaining feature correspondence across frames.

    Manages feature tracks and determines when features should be marginalized.
    """

    def __init__(self,
                 max_track_length: int = 20,
                 min_track_length: int = 3,
                 max_features: int = 200):
        """
        Initialize feature tracker.

        Args:
            max_track_length: Maximum number of frames to track a feature
            min_track_length: Minimum track length before feature can be used
            max_features: Maximum number of features to track
        """
        self.max_track_length = max_track_length
        self.min_track_length = min_track_length
        self.max_features = max_features

        # Active tracks: feature_id -> Feature
        self.active_tracks: Dict[int, Feature] = {}

        # Feature counter
        self._feature_id_counter = 0

        # Frame counter
        self._frame_id_counter = 0

    def add_features(self, features: List[Tuple[float, float]], camera_state_id: int) -> List[Feature]:
        """
        Add new features detected in current frame.

        Args:
            features: List of (u, v) normalized coordinates
            camera_state_id: ID of camera state

        Returns:
            List of Feature objects
        """
        new_features = []

        for u, v in features:
            if len(self.active_tracks) >= self.max_features:
                break

            feature = Feature(
                feature_id=self._feature_id_counter,
                u=u,
                v=v,
                camera_id=0
            )
            feature.add_observation(camera_state_id, np.array([u, v]))

            self.active_tracks[self._feature_id_counter] = feature
            new_features.append(feature)

            self._feature_id_counter += 1

        return new_features

    def track_features(self,
                      prev_features: List[Feature],
                      curr_observations: List[Tuple[int, float, float]],
                      camera_state_id: int) -> Tuple[List[Feature], List[Feature]]:
        """
        Track features from previous frame to current frame.

        Args:
            prev_features: Features from previous frame
            curr_observations: List of (feature_id, u, v) in current frame
            camera_state_id: ID of current camera state

        Returns:
            (tracked_features, lost_features)
        """
        tracked = []
        lost = []

        # Build lookup for current observations
        curr_obs_dict = {fid: (u, v) for fid, u, v in curr_observations}

        for feature in prev_features:
            if feature.feature_id in curr_obs_dict:
                # Feature tracked successfully
                u, v = curr_obs_dict[feature.feature_id]
                feature.u = u
                feature.v = v
                feature.add_observation(camera_state_id, np.array([u, v]))

                # Check if track is too long
                if feature.num_observations >= self.max_track_length:
                    feature.status = FeatureStatus.READY_FOR_UPDATE
                else:
                    feature.status = FeatureStatus.TRACKED

                tracked.append(feature)
            else:
                # Feature lost
                feature.status = FeatureStatus.LOST
                lost.append(feature)

                # Remove from active tracks
                if feature.feature_id in self.active_tracks:
                    del self.active_tracks[feature.feature_id]

        return tracked, lost

    def get_features_for_update(self) -> List[Feature]:
        """
        Get features that are ready for MSCKF update.

        Returns:
            List of features that can be used for update
        """
        ready_features = []

        for feature in list(self.active_tracks.values()):
            # Features that are lost or reached max track length
            if (feature.status == FeatureStatus.READY_FOR_UPDATE or
                feature.status == FeatureStatus.LOST):

                # Check minimum track length
                if feature.num_observations >= self.min_track_length:
                    ready_features.append(feature)
                    feature.status = FeatureStatus.MARGINALIZED

                # Remove from active tracks
                if feature.feature_id in self.active_tracks:
                    del self.active_tracks[feature.feature_id]

        return ready_features

    def prune_tracks(self, valid_camera_state_ids: List[int]):
        """
        Remove observations from camera states that no longer exist.

        Args:
            valid_camera_state_ids: List of valid camera state IDs
        """
        valid_ids_set = set(valid_camera_state_ids)

        for feature in list(self.active_tracks.values()):
            # Filter observations
            feature.observations = [
                (cam_id, obs) for cam_id, obs in feature.observations
                if cam_id in valid_ids_set
            ]

            # Remove feature if no observations remain
            if len(feature.observations) == 0:
                if feature.feature_id in self.active_tracks:
                    del self.active_tracks[feature.feature_id]

    def reset(self):
        """Reset tracker state."""
        self.active_tracks.clear()
        self._feature_id_counter = 0
        self._frame_id_counter = 0

    def get_active_features(self) -> List[Feature]:
        """Get all active features."""
        return list(self.active_tracks.values())

    @property
    def num_active_tracks(self) -> int:
        """Number of currently active tracks."""
        return len(self.active_tracks)


class OpticalFlowTracker:
    """
    Optical flow-based feature tracker.

    This is a simplified tracker interface. In practice, you would integrate
    with OpenCV's optical flow or similar algorithms.
    """

    def __init__(self,
                 window_size: Tuple[int, int] = (21, 21),
                 max_level: int = 3,
                 min_eigen_threshold: float = 0.001):
        """
        Initialize optical flow tracker.

        Args:
            window_size: Size of search window
            max_level: Maximum pyramid level
            min_eigen_threshold: Minimum eigenvalue threshold
        """
        self.window_size = window_size
        self.max_level = max_level
        self.min_eigen_threshold = min_eigen_threshold

    def track(self,
             prev_image: np.ndarray,
             curr_image: np.ndarray,
             prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track points from previous image to current image.

        Args:
            prev_image: Previous image (grayscale)
            curr_image: Current image (grayscale)
            prev_points: Points to track (N, 2)

        Returns:
            (curr_points, status) where status indicates successful tracking
        """
        # This is a placeholder. In practice, use cv2.calcOpticalFlowPyrLK
        # or similar algorithm

        # For now, return identity (no motion)
        curr_points = prev_points.copy()
        status = np.ones(len(prev_points), dtype=bool)

        return curr_points, status


class FeatureDetector:
    """
    Feature detector interface.

    In practice, integrate with FAST, ORB, SIFT, etc.
    """

    def __init__(self,
                 detector_type: str = "FAST",
                 num_features: int = 200,
                 quality_level: float = 0.01,
                 min_distance: float = 10.0):
        """
        Initialize feature detector.

        Args:
            detector_type: Type of detector (FAST, ORB, SIFT, etc.)
            num_features: Maximum number of features to detect
            quality_level: Quality threshold for corner detection
            min_distance: Minimum distance between features
        """
        self.detector_type = detector_type
        self.num_features = num_features
        self.quality_level = quality_level
        self.min_distance = min_distance

    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect features in image.

        Args:
            image: Input image (grayscale)
            mask: Optional mask for detection region

        Returns:
            Array of detected points (N, 2)
        """
        # Placeholder implementation
        # In practice, use cv2.goodFeaturesToTrack, cv2.FAST, etc.

        # Return empty array
        return np.zeros((0, 2))

    def detect_grid(self,
                   image: np.ndarray,
                   grid_size: Tuple[int, int] = (5, 5),
                   features_per_cell: int = 5) -> np.ndarray:
        """
        Detect features uniformly distributed across image grid.

        Args:
            image: Input image
            grid_size: Grid dimensions (rows, cols)
            features_per_cell: Features to detect per cell

        Returns:
            Array of detected points (N, 2)
        """
        height, width = image.shape[:2]
        cell_height = height // grid_size[0]
        cell_width = width // grid_size[1]

        all_points = []

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Define cell region
                y_start = i * cell_height
                y_end = (i + 1) * cell_height if i < grid_size[0] - 1 else height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width if j < grid_size[1] - 1 else width

                # Create mask for this cell
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y_start:y_end, x_start:x_end] = 255

                # Detect features in cell
                cell_points = self.detect(image, mask)

                # Limit number of features per cell
                if len(cell_points) > features_per_cell:
                    cell_points = cell_points[:features_per_cell]

                all_points.append(cell_points)

        if all_points:
            return np.vstack(all_points)
        else:
            return np.zeros((0, 2))
