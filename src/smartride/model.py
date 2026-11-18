"""Pickup ETA regression baseline."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

FEATURES = ["distance_km", "hour", "traffic_index", "weather_index"]


def train_eta_model(trips, random_state: int = 42):
    """Fit a reproducible baseline and return model plus holdout metrics."""
    X = trips[FEATURES]
    y = trips["pickup_eta_min"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=180, min_samples_leaf=3, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae_min": float(mean_absolute_error(y_test, pred)),
        "rmse_min": float(root_mean_squared_error(y_test, pred)),
    }
    return model, metrics
