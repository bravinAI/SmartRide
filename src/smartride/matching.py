"""Rider-driver assignment using predicted pickup ETAs."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .features import pairwise_distance_matrix


def build_eta_matrix(riders, drivers, model, hour=17, traffic_index=0.5, weather_index=0.2):
    """Predict ETA for every rider-driver pair; output shape (riders, drivers)."""
    distances = pairwise_distance_matrix(riders, drivers)
    rows = []
    for i in range(len(riders)):
        for j in range(len(drivers)):
            rows.append({"distance_km": distances[i, j], "hour": hour,
                         "traffic_index": traffic_index, "weather_index": weather_index})
    return model.predict(pd.DataFrame(rows)).reshape(len(riders), len(drivers))


def optimize_assignments(eta_matrix, riders, drivers):
    """Minimize total predicted pickup ETA with one-to-one batch assignments."""
    if eta_matrix.shape != (len(riders), len(drivers)):
        raise ValueError("ETA matrix shape must be (number of riders, number of drivers)")
    rider_idx, driver_idx = linear_sum_assignment(eta_matrix)
    return pd.DataFrame({
        "rider_id": riders.iloc[rider_idx]["rider_id"].to_numpy(),
        "driver_id": drivers.iloc[driver_idx]["driver_id"].to_numpy(),
        "predicted_pickup_min": eta_matrix[rider_idx, driver_idx],
        "pickup_distance_km": pairwise_distance_matrix(riders, drivers)[rider_idx, driver_idx],
    }).sort_values("rider_id").reset_index(drop=True)


def greedy_nearest_assignments(eta_matrix, riders, drivers):
    """Simple comparison baseline: assign each rider the fastest remaining driver."""
    remaining = set(range(len(drivers)))
    pairs = []
    for i in range(len(riders)):
        if not remaining:
            break
        j = min(remaining, key=lambda col: eta_matrix[i, col])
        remaining.remove(j)
        pairs.append((i, j))
    return pd.DataFrame({
        "rider_id": [riders.iloc[i]["rider_id"] for i, _ in pairs],
        "driver_id": [drivers.iloc[j]["driver_id"] for _, j in pairs],
        "predicted_pickup_min": [eta_matrix[i, j] for i, j in pairs],
    })
