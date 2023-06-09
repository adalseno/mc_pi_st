import numpy as np
import pandas as pd
from numba import njit
import streamlit as st

@njit(cache=True)
def pi_sim(N=100,seed:int=None)->float:
    if seed is not None:
        np.random.seed = seed
    x = np.random.uniform(0,1,size=N)
    y = np.sqrt(1-x**2)
    return 4.0 * np.mean(y)

@njit(cache=True)
def pi_mc(N:int=100)->np.ndarray[float]:
    pi = np.zeros((N))
    for i in range(N):
        pi[i]=pi_sim()
    return pi

# Title and subtitle
st.title("Monte Carlo simulation")
st.subheader(
    "Compute $\pi$ value using a Monte Carlo simulation"
)

# Select number of trials from listbox
trial_range = map(lambda x: int(np.power(10, x)), range(2, 8))  # from 100 to 1_000_000

# I used a comma to seprate thousands for better readability
trials = st.selectbox(
    "Select number of trials", trial_range, format_func=lambda x: f"{x:,}"
)

# Get data
data_load_state = st.text("Computing data...")
data = pi_mc(trials)
data_load_state.text("Done!")

# Add statistics
avg = np.mean(data)  # Mean
sigma = np.std(data)  # Std
lowb = avg - 1.96 * sigma / np.sqrt(trials)  # Low Bound
upb = avg + 1.96 * sigma / np.sqrt(trials)  # Upper Bound
width = upb - lowb
confidence = 0.95
lowp, upp = np.percentile(data,[100*(1-confidence)/2,100*(1-(1-confidence)/2)],method="hazen")
width_p = upp - lowp 

# Create DataFrame with statistics
stat_df = pd.DataFrame(
    {
        "Trials": f"{trials:,}",
        "Average": f"{avg:.4f}",
        "St. Dev": f"{sigma:.4f}",
        "Low Bound": f"{lowb:.4f}",
        "Upper Bound": f"{upb:.4f}",
        "Width": f" {width:.4f}",
        "2.5 perc.":f"{lowp:.4f}",
        "97.5 perc.":f"{upp:.4f}",
        "Width_p":f"{width_p:.4f}"
    },
    index=["Monte Carlo Simulation"],
)


# Get bounds
bound_min = data.min().round(4)
bound_max = data.max().round(4)

# Get data for histogram
y, bins = np.histogram(data, bins=24, density=True)
bins = np.round(bins,3)

# Create Series for better management
hist_values = pd.Series(y, index=bins[:-1], name="Estimated pi value")

# Display statistics (the values are strings)
st.table(stat_df)

# Plot histogram
st.bar_chart(hist_values)
st.text(f"Real value up to 4 decimal places: {np.pi:.4f}")
