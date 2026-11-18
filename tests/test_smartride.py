import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from smartride.data import generate_marketplace
from smartride.features import haversine_km, pairwise_distance_matrix
from smartride.model import train_eta_model
from smartride.matching import build_eta_matrix, optimize_assignments


def test_haversine_same_point_is_zero():
    assert float(haversine_km(0, 0, 0, 0)) == pytest.approx(0.0)


def test_distance_matrix_shape():
    riders, drivers, _ = generate_marketplace(3, 4)
    assert pairwise_distance_matrix(riders, drivers).shape == (3, 4)


def test_model_and_matching_assign_each_rider_once():
    riders, drivers, trips = generate_marketplace(5, 7)
    model, metrics = train_eta_model(trips)
    matrix = build_eta_matrix(riders, drivers, model)
    assignments = optimize_assignments(matrix, riders, drivers)
    assert matrix.shape == (5, 7)
    assert len(assignments) == 5
    assert assignments.rider_id.nunique() == 5
    assert assignments.driver_id.nunique() == 5
    assert metrics["mae_min"] >= 0


def test_invalid_marketplace_size():
    with pytest.raises(ValueError):
        generate_marketplace(0, 2)
