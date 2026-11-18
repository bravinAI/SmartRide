"""Create reproducible synthetic riders, drivers and pickup observations."""
from __future__ import annotations
import numpy as np
import pandas as pd

# Approximate Nairobi-area centre. These are synthetic coordinates around a
# generic centre, not actual user or driver locations.
CENTRE_LAT = -1.2864
CENTRE_LON = 36.8172


def generate_marketplace(n_riders: int = 12, n_drivers: int = 15, seed: int = 42):
    """Return (riders, drivers, historical_trips) using synthetic data."""
    if n_riders < 1 or n_drivers < 1:
        raise ValueError("n_riders and n_drivers must both be at least 1")
    rng = np.random.default_rng(seed)

    def locations(n, prefix):
        return pd.DataFrame({
            f"{prefix}_id": [f"{prefix.upper()}{i+1:03d}" for i in range(n)],
            "lat": CENTRE_LAT + rng.normal(0, 0.035, n),
            "lon": CENTRE_LON + rng.normal(0, 0.035, n),
        })

    riders = locations(n_riders, "rider")
    drivers = locations(n_drivers, "driver")
    n = 2500
    distance = rng.uniform(0.2, 12.0, n)
    hour = rng.integers(0, 24, n)
    traffic = rng.uniform(0, 1, n)
    weather = rng.uniform(0, 1, n)
    rush = np.where(((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19)), 1.0, 0.0)
    # A transparent synthetic data-generating process with noise.
    eta = (1.8 + 2.1 * distance + 5.0 * traffic + 1.7 * weather
           + 2.2 * rush + rng.normal(0, 1.3, n))
    eta = np.clip(eta, 1.0, None)
    trips = pd.DataFrame({"distance_km": distance, "hour": hour,
                          "traffic_index": traffic, "weather_index": weather,
                          "pickup_eta_min": eta})
    return riders, drivers, trips
