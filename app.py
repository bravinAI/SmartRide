"""Interactive SmartRide demonstration dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import streamlit as st
from smartride.data import generate_marketplace
from smartride.model import train_eta_model
from smartride.matching import build_eta_matrix, optimize_assignments, greedy_nearest_assignments

st.set_page_config(page_title="SmartRide | Intelligent Matching", page_icon="🚗", layout="wide")
st.title("🚗 SmartRide")
st.caption("An educational ride-matching simulator — synthetic data, not a live ride service.")
with st.sidebar:
    st.header("Simulation settings")
    n_riders = st.slider("Riders requesting rides", 2, 30, 12)
    n_drivers = st.slider("Available drivers", 2, 35, 15)
    hour = st.slider("Hour of day", 0, 23, 17)
    traffic = st.slider("Traffic index", 0.0, 1.0, 0.5, 0.05)
    weather = st.slider("Weather index", 0.0, 1.0, 0.2, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)

riders, drivers, trips = generate_marketplace(n_riders, n_drivers, int(seed))
model, metrics = train_eta_model(trips)
eta = build_eta_matrix(riders, drivers, model, hour, traffic, weather)
optimal = optimize_assignments(eta, riders, drivers)
greedy = greedy_nearest_assignments(eta, riders, drivers)

c1, c2, c3 = st.columns(3)
c1.metric("Riders", n_riders)
c2.metric("Available drivers", n_drivers)
c3.metric("ETA model MAE", f"{metrics['mae_min']:.2f} min")
st.subheader("Batch assignment")
left, right = st.columns(2)
with left:
    st.markdown("#### Optimized assignment")
    st.dataframe(optimal.round(2), use_container_width=True, hide_index=True)
    st.metric("Total predicted pickup time", f"{optimal.predicted_pickup_min.sum():.1f} min")
with right:
    st.markdown("#### Greedy rider-by-rider baseline")
    st.dataframe(greedy.round(2), use_container_width=True, hide_index=True)
    st.metric("Total predicted pickup time", f"{greedy.predicted_pickup_min.sum():.1f} min")

st.subheader("How to interpret this")
st.write("The Random Forest predicts pickup time from synthetic distance, hour, traffic and weather features. The Hungarian algorithm then minimizes the total predicted pickup time across this batch, enforcing one driver per assigned rider. It is an optimization algorithm, not another ML model.")
st.info("This demo uses straight-line geographic distance and synthetic traffic. Real-world dispatch needs road-network ETAs, service constraints, safety, privacy, fairness and marketplace-wide objectives.")
