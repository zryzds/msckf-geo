"""
Unit tests for geographic modules.
"""
import pytest
import numpy as np
from src.geographic.geo_features import GeoFeatureManager, GeoConstraint
from src.geographic.measurements import (
    GPSMeasurement,
    GPSProcessor,
    GeodeticUtils,
    GPSOutlierDetector
)
from src.core.state import create_initial_state


class TestGeoFeatureManager:
    """Tests for geographic feature manager."""

    @pytest.fixture
    def manager(self):
        return GeoFeatureManager(
            ref_latitude=37.7749,
            ref_longitude=-122.4194,
            ref_altitude=10.0
        )

    def test_creation(self, manager):
        assert manager.ref_lat == 37.7749
        assert len(manager.geo_features) == 0

    def test_add_feature(self, manager):
        feature_id = manager.add_geo_feature(37.7750, -122.4195, 12.0)
        assert feature_id == 0
        assert len(manager.geo_features) == 1

    def test_get_feature(self, manager):
        feature_id = manager.add_geo_feature(37.7750, -122.4195, 12.0)
        feature = manager.get_geo_feature(feature_id)
        assert feature is not None
        assert feature.latitude == 37.7750

    def test_nearby_features(self, manager):
        # Add features
        manager.add_geo_feature(37.7750, -122.4195, 12.0)
        manager.add_geo_feature(37.7800, -122.4200, 15.0)  # Far away
        manager.add_geo_feature(37.7751, -122.4196, 13.0)  # Nearby

        # Query near first feature
        nearby = manager.get_nearby_features(37.7750, -122.4195, radius=200.0)

        # Should find at least 2 nearby features
        assert len(nearby) >= 2

    def test_to_enu(self, manager):
        feature_id = manager.add_geo_feature(
            manager.ref_lat,
            manager.ref_lon,
            manager.ref_alt
        )
        feature = manager.get_geo_feature(feature_id)
        enu = manager.to_enu(feature)

        # Feature at reference should be at origin
        assert np.allclose(enu, np.zeros(3), atol=1.0)


class TestGPSProcessor:
    """Tests for GPS processor."""

    @pytest.fixture
    def processor(self):
        return GPSProcessor(
            ref_latitude=37.7749,
            ref_longitude=-122.4194,
            ref_altitude=10.0
        )

    def test_creation(self, processor):
        assert processor.base_position_noise == 5.0

    def test_process_measurement(self, processor):
        gps_meas = GPSMeasurement(
            timestamp=1.0,
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0,
            horizontal_accuracy=5.0
        )

        position_enu, covariance = processor.process_measurement(gps_meas)

        assert position_enu.shape == (3,)
        assert covariance.shape == (3, 3)

    def test_measurement_model(self, processor):
        state = create_initial_state()

        gps_meas = GPSMeasurement(
            timestamp=1.0,
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0
        )

        H, R, residual = processor.compute_measurement_model(state, gps_meas)

        assert H.shape[0] == 3  # Position measurement
        assert H.shape[1] == state.state_dim
        assert R.shape == (3, 3)
        assert residual.shape == (3,)

    def test_quality_check(self, processor):
        # Good measurement
        good_meas = GPSMeasurement(
            timestamp=1.0,
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0,
            num_satellites=8,
            horizontal_accuracy=3.0
        )
        assert processor.check_measurement_quality(good_meas)

        # Poor measurement (few satellites)
        bad_meas = GPSMeasurement(
            timestamp=1.0,
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0,
            num_satellites=2,
            horizontal_accuracy=3.0
        )
        assert not processor.check_measurement_quality(bad_meas)


class TestGeodeticUtils:
    """Tests for geodetic utilities."""

    def test_haversine_distance(self):
        # Distance from San Francisco to Los Angeles (approx 559 km)
        lat1, lon1 = 37.7749, -122.4194  # SF
        lat2, lon2 = 34.0522, -118.2437  # LA

        distance = GeodeticUtils.haversine_distance(lat1, lon1, lat2, lon2)

        # Should be around 559 km
        assert 550000 < distance < 570000

    def test_zero_distance(self):
        lat, lon = 37.7749, -122.4194
        distance = GeodeticUtils.haversine_distance(lat, lon, lat, lon)
        assert np.isclose(distance, 0.0, atol=1.0)

    def test_bearing(self):
        # Bearing from SF to point directly north
        lat1, lon1 = 37.0, -122.0
        lat2, lon2 = 38.0, -122.0

        bearing = GeodeticUtils.compute_bearing(lat1, lon1, lat2, lon2)

        # Should be approximately 0 degrees (north)
        assert np.isclose(bearing, 0.0, atol=1.0) or np.isclose(bearing, 360.0, atol=1.0)

    def test_destination_point(self):
        lat, lon = 37.0, -122.0
        bearing = 0.0  # North
        distance = 1000.0  # 1 km

        lat2, lon2 = GeodeticUtils.destination_point(lat, lon, bearing, distance)

        # Latitude should increase (moving north)
        assert lat2 > lat
        # Longitude should stay approximately same
        assert np.isclose(lon2, lon, atol=0.01)


class TestGPSOutlierDetector:
    """Tests for GPS outlier detector."""

    def test_first_measurement(self):
        detector = GPSOutlierDetector()

        gps_meas = GPSMeasurement(
            timestamp=1.0,
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0
        )

        position_enu = np.array([100.0, 200.0, 12.0])

        # First measurement should always be accepted
        assert detector.check_measurement(gps_meas, position_enu)

    def test_outlier_detection_velocity(self):
        detector = GPSOutlierDetector(max_velocity=10.0)  # 10 m/s max

        # First measurement
        gps_meas1 = GPSMeasurement(
            timestamp=1.0,
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0
        )
        position1 = np.array([0.0, 0.0, 12.0])
        detector.check_measurement(gps_meas1, position1)

        # Second measurement - unrealistic jump
        gps_meas2 = GPSMeasurement(
            timestamp=2.0,  # 1 second later
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0
        )
        position2 = np.array([100.0, 0.0, 12.0])  # 100m in 1s = 100 m/s

        # Should be rejected (exceeds max velocity)
        assert not detector.check_measurement(gps_meas2, position2)

    def test_outlier_detection_jump(self):
        detector = GPSOutlierDetector(max_jump=10.0)  # 10m max jump

        # First measurement
        gps_meas1 = GPSMeasurement(
            timestamp=1.0,
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0
        )
        position1 = np.array([0.0, 0.0, 12.0])
        detector.check_measurement(gps_meas1, position1)

        # Second measurement - large jump
        gps_meas2 = GPSMeasurement(
            timestamp=1.1,  # 0.1 seconds later
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0
        )
        position2 = np.array([100.0, 0.0, 12.0])  # 100m jump

        # Should be rejected (exceeds max jump)
        assert not detector.check_measurement(gps_meas2, position2)

    def test_reset(self):
        detector = GPSOutlierDetector()

        gps_meas = GPSMeasurement(
            timestamp=1.0,
            latitude=37.7750,
            longitude=-122.4195,
            altitude=12.0
        )
        position = np.array([0.0, 0.0, 12.0])
        detector.check_measurement(gps_meas, position)

        # Reset
        detector.reset()

        assert detector.last_position is None
        assert detector.last_timestamp is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
