# SmartRide 🚗

**An educational, Uber-inspired ride matching system** built with Python, pandas, scikit-learn and assignment optimization.

SmartRide demonstrates how a ride-hailing marketplace can estimate driver pickup times and assign multiple riders to different drivers. It uses **synthetic demonstration data only**; it is not connected to Uber, real drivers, live maps, payments or a production dispatch network.

## What it demonstrates

1. Generate reproducible synthetic riders and available drivers.
2. Calculate approximate geographic distances with the Haversine formula.
3. Train a regression model to estimate pickup ETA.
4. Build a rider–driver cost matrix from predicted ETAs.
5. Use the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to find a globally efficient one-to-one assignment for the current batch.
6. Compare optimized matching with a simple nearest-driver baseline.
7. Explore the result in a Streamlit dashboard.

> Important distinction: the ETA model is machine learning. The Hungarian assignment algorithm is mathematical optimization, not machine learning. Real ride-hailing systems combine prediction and optimization, plus many operational rules.

## Quick start

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/bravinAI/SmartRide.git
cd SmartRide
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter lab
```

Open `notebooks/01_smartride_walkthrough.ipynb` and run the cells from top to bottom. The notebook explains each step and prints the model metrics and assignments.

### Run the dashboard

```bash
streamlit run app.py
```

## Project structure

```text
SmartRide/
├── app.py
├── data/                  # Generated data is written here when requested
├── notebooks/
│   └── 01_smartride_walkthrough.ipynb
├── src/smartride/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── matching.py
├── tests/
│   └── test_smartride.py
├── requirements.txt
└── README.md
```

## How the matching works

For each available rider–driver pair, SmartRide estimates the driver's pickup ETA. These predictions form a cost matrix. The Hungarian algorithm selects a set of one-to-one pairs that minimizes the **sum of predicted pickup ETAs** for the batch. Unmatched riders or drivers can occur when the two groups have different sizes.

This is a deliberately simplified objective. A real platform may consider acceptance probability, cancellations, fairness, driver earnings, vehicle class, road-network ETA, batching latency, service constraints and marketplace-wide effects. SmartRide does not claim to reproduce Uber's proprietary production system.

## Model and evaluation

- Model: scikit-learn `RandomForestRegressor` as an accessible baseline.
- Target: synthetic pickup time in minutes.
- Features: straight-line distance, hour, traffic index and weather index.
- Evaluation: MAE and RMSE on a held-out test split.
- Reproducibility: fixed random seeds.

Because the data is synthetic, metrics demonstrate a workflow rather than real-world predictive performance. The straight-line distance is not road distance; the dashboard labels it accordingly.

## Tests

```bash
pytest -q
```

## Learning roadmap

- [x] Synthetic marketplace data
- [x] Geographic distance feature
- [x] ETA regression baseline
- [x] Batch assignment optimization
- [x] Nearest-driver baseline comparison
- [x] Interactive dashboard
- [ ] Replace synthetic data with a documented public dataset
- [ ] Add road-network routing and time-aware traffic features
- [ ] Add acceptance/cancellation modelling and fairness diagnostics
- [ ] Package, monitor and deploy the model

## Disclaimer

SmartRide is an independent educational portfolio project and is not affiliated with, endorsed by, or connected to Uber. No personal or real-time location data is collected.