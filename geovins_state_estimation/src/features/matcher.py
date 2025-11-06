"""
Feature matching module.

Provides feature matching algorithms for establishing correspondences.
"""
import numpy as np
from typing import List, Tuple, Optional
from ..core.types import Feature


class FeatureMatcher:
    """
    Feature matcher using descriptor similarity.

    Supports various matching strategies including brute force and FLANN.
    """

    def __init__(self,
                 match_threshold: float = 0.7,
                 use_ratio_test: bool = True,
                 ratio_threshold: float = 0.8):
        """
        Initialize feature matcher.

        Args:
            match_threshold: Distance threshold for matches
            use_ratio_test: Use Lowe's ratio test
            ratio_threshold: Ratio for Lowe's ratio test
        """
        self.match_threshold = match_threshold
        self.use_ratio_test = use_ratio_test
        self.ratio_threshold = ratio_threshold

    def match_features(self,
                      features1: List[Feature],
                      features2: List[Feature],
                      descriptors1: Optional[np.ndarray] = None,
                      descriptors2: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Match features between two sets.

        Args:
            features1: Features from first set
            features2: Features from second set
            descriptors1: Optional descriptors for features1
            descriptors2: Optional descriptors for features2

        Returns:
            List of matches (idx1, idx2)
        """
        if descriptors1 is None or descriptors2 is None:
            # Fall back to coordinate-based matching
            return self._match_by_coordinates(features1, features2)

        matches = []

        for i, desc1 in enumerate(descriptors1):
            # Compute distances to all features in set 2
            distances = np.linalg.norm(descriptors2 - desc1, axis=1)

            # Find best and second-best matches
            sorted_indices = np.argsort(distances)
            best_idx = sorted_indices[0]
            best_dist = distances[best_idx]

            if self.use_ratio_test and len(sorted_indices) > 1:
                second_best_dist = distances[sorted_indices[1]]
                ratio = best_dist / second_best_dist

                if ratio < self.ratio_threshold and best_dist < self.match_threshold:
                    matches.append((i, best_idx))
            else:
                if best_dist < self.match_threshold:
                    matches.append((i, best_idx))

        return matches

    def _match_by_coordinates(self,
                             features1: List[Feature],
                             features2: List[Feature],
                             max_distance: float = 20.0) -> List[Tuple[int, int]]:
        """
        Match features based on coordinate proximity.

        Args:
            features1: Features from first set
            features2: Features from second set
            max_distance: Maximum pixel distance for match

        Returns:
            List of matches (idx1, idx2)
        """
        matches = []

        for i, f1 in enumerate(features1):
            min_dist = float('inf')
            best_match = -1

            for j, f2 in enumerate(features2):
                dist = np.sqrt((f1.u - f2.u)**2 + (f1.v - f2.v)**2)

                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    best_match = j

            if best_match >= 0:
                matches.append((i, best_match))

        return matches


class EpipolarMatcher:
    """
    Feature matcher using epipolar geometry constraints.

    Matches features along epipolar lines between views.
    """

    def __init__(self,
                 epipolar_threshold: float = 1.0,
                 descriptor_threshold: float = 0.7):
        """
        Initialize epipolar matcher.

        Args:
            epipolar_threshold: Maximum distance from epipolar line (pixels)
            descriptor_threshold: Descriptor distance threshold
        """
        self.epipolar_threshold = epipolar_threshold
        self.descriptor_threshold = descriptor_threshold

    def match_with_fundamental(self,
                               features1: List[Feature],
                               features2: List[Feature],
                               F: np.ndarray,
                               descriptors1: Optional[np.ndarray] = None,
                               descriptors2: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Match features using fundamental matrix constraint.

        Args:
            features1: Features from first image
            features2: Features from second image
            F: Fundamental matrix (3x3)
            descriptors1: Optional descriptors for features1
            descriptors2: Optional descriptors for features2

        Returns:
            List of matches (idx1, idx2)
        """
        matches = []

        for i, f1 in enumerate(features1):
            # Compute epipolar line in second image
            p1 = np.array([f1.u, f1.v, 1.0])
            epipolar_line = F @ p1  # l = F @ p1

            # Find features in image 2 close to epipolar line
            candidates = []

            for j, f2 in enumerate(features2):
                p2 = np.array([f2.u, f2.v, 1.0])

                # Distance from point to line: |ax + by + c| / sqrt(a^2 + b^2)
                distance = abs(epipolar_line @ p2) / np.sqrt(
                    epipolar_line[0]**2 + epipolar_line[1]**2
                )

                if distance < self.epipolar_threshold:
                    candidates.append(j)

            # Match with best descriptor match among candidates
            if len(candidates) > 0 and descriptors1 is not None and descriptors2 is not None:
                desc1 = descriptors1[i]
                candidate_descs = descriptors2[candidates]

                distances = np.linalg.norm(candidate_descs - desc1, axis=1)
                best_idx = np.argmin(distances)

                if distances[best_idx] < self.descriptor_threshold:
                    matches.append((i, candidates[best_idx]))

        return matches

    @staticmethod
    def compute_fundamental_matrix(
        points1: np.ndarray,
        points2: np.ndarray,
        method: str = "8point"
    ) -> Optional[np.ndarray]:
        """
        Compute fundamental matrix from point correspondences.

        Args:
            points1: Points in first image (N, 2)
            points2: Points in second image (N, 2)
            method: Method to use ("8point" or "ransac")

        Returns:
            Fundamental matrix (3x3) or None if computation fails
        """
        if len(points1) < 8:
            return None

        # Normalize points
        points1_norm, T1 = EpipolarMatcher._normalize_points(points1)
        points2_norm, T2 = EpipolarMatcher._normalize_points(points2)

        # Build constraint matrix
        N = len(points1)
        A = np.zeros((N, 9))

        for i in range(N):
            x1, y1 = points1_norm[i]
            x2, y2 = points2_norm[i]

            A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)

        # Enforce rank-2 constraint
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ Vt

        # Denormalize
        F = T2.T @ F @ T1

        return F

    @staticmethod
    def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize points for numerical stability.

        Args:
            points: Points (N, 2)

        Returns:
            (normalized_points, transformation_matrix)
        """
        centroid = np.mean(points, axis=0)
        shifted = points - centroid

        mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
        scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0

        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])

        # Transform points
        points_homog = np.column_stack([points, np.ones(len(points))])
        points_norm = (T @ points_homog.T).T
        points_norm = points_norm[:, :2] / points_norm[:, 2:3]

        return points_norm, T


class RANSACMatcher:
    """
    RANSAC-based feature matcher for outlier rejection.
    """

    def __init__(self,
                 ransac_threshold: float = 3.0,
                 max_iterations: int = 1000,
                 confidence: float = 0.99):
        """
        Initialize RANSAC matcher.

        Args:
            ransac_threshold: Inlier threshold (pixels)
            max_iterations: Maximum RANSAC iterations
            confidence: Desired confidence level
        """
        self.ransac_threshold = ransac_threshold
        self.max_iterations = max_iterations
        self.confidence = confidence

    def refine_matches(self,
                      matches: List[Tuple[int, int]],
                      features1: List[Feature],
                      features2: List[Feature]) -> List[Tuple[int, int]]:
        """
        Refine matches using RANSAC.

        Args:
            matches: Initial matches
            features1: Features from first image
            features2: Features from second image

        Returns:
            Filtered matches (inliers only)
        """
        if len(matches) < 8:
            return matches

        # Extract matched points
        points1 = np.array([[features1[m[0]].u, features1[m[0]].v] for m in matches])
        points2 = np.array([[features2[m[1]].u, features2[m[1]].v] for m in matches])

        # RANSAC loop
        best_inliers = []
        best_F = None

        for _ in range(self.max_iterations):
            # Sample 8 random matches
            sample_indices = np.random.choice(len(matches), 8, replace=False)
            sample_points1 = points1[sample_indices]
            sample_points2 = points2[sample_indices]

            # Compute fundamental matrix
            F = EpipolarMatcher.compute_fundamental_matrix(
                sample_points1,
                sample_points2,
                method="8point"
            )

            if F is None:
                continue

            # Compute inliers
            inliers = self._compute_inliers(points1, points2, F)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F

                # Early termination check
                inlier_ratio = len(inliers) / len(matches)
                if self._check_termination(inlier_ratio, len(inliers)):
                    break

        # Return inlier matches
        if len(best_inliers) > 0:
            return [matches[i] for i in best_inliers]
        else:
            return matches

    def _compute_inliers(self,
                        points1: np.ndarray,
                        points2: np.ndarray,
                        F: np.ndarray) -> List[int]:
        """Compute inliers for given fundamental matrix."""
        inliers = []

        for i, (p1, p2) in enumerate(zip(points1, points2)):
            p1_h = np.array([p1[0], p1[1], 1.0])
            p2_h = np.array([p2[0], p2[1], 1.0])

            # Symmetric epipolar distance
            error1 = abs(p2_h @ F @ p1_h) / np.sqrt((F @ p1_h)[0]**2 + (F @ p1_h)[1]**2)
            error2 = abs(p1_h @ F.T @ p2_h) / np.sqrt((F.T @ p2_h)[0]**2 + (F.T @ p2_h)[1]**2)

            error = (error1 + error2) / 2

            if error < self.ransac_threshold:
                inliers.append(i)

        return inliers

    def _check_termination(self, inlier_ratio: float, num_inliers: int) -> bool:
        """Check if RANSAC should terminate early."""
        if num_inliers < 8:
            return False

        # Compute number of iterations needed for desired confidence
        epsilon = 1 - inlier_ratio
        if epsilon >= 1:
            return False

        num_iterations_needed = np.log(1 - self.confidence) / np.log(1 - (1 - epsilon)**8)

        return self.max_iterations > num_iterations_needed
