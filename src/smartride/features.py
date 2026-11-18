"""Geographic feature engineering."""
import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in kilometres (not road distance)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def pairwise_distance_matrix(riders: pd.DataFrame, drivers: pd.DataFrame) -> np.ndarray:
    """Rows are riders; columns are drivers."""
    return haversine_km(
        riders["lat"].to_numpy()[:, None], riders["lon"].to_numpy()[:, None],
        drivers["lat"].to_numpy()[None, :], drivers["lon"].to_numpy()[None, :]
    )
