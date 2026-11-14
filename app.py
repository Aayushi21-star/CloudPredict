
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="CloudPredict",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ CloudPredict")
st.subheader("Intelligent Storage Forecasting System")
st.write(
    "Analyze historical storage usage and predict future storage requirements."
)

# Generate sample simulated dataset
dates = pd.date_range(start="2026-01-01", periods=30, freq="D")

usage = [
    100, 105, 108, 112, 115, 119, 121, 125, 128, 132,
    135, 138, 142, 145, 149, 152, 156, 160, 163, 167,
    170, 174, 178, 181, 185, 189, 192, 196, 200, 204
]

sample_data = pd.DataFrame({
    "Date": dates,
    "Storage_Used_GB": usage
})

# Sidebar settings
st.sidebar.header("Project Settings")

capacity = st.sidebar.number_input(
    "Total Storage Capacity (GB)",
    min_value=100,
    value=300
)

threshold = st.sidebar.slider(
    "Warning Threshold (%)",
    min_value=50,
    max_value=100,
    value=90
)

days = st.sidebar.selectbox(
    "Forecast Days",
    [7, 15, 30]
)

# CSV upload
uploaded_file = st.file_uploader(
    "Upload Storage CSV (optional)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if not {"Date", "Storage_Used_GB"}.issubset(data.columns):
            st.error(
                "CSV must contain Date and Storage_Used_GB columns."
            )
            st.stop()

        data["Date"] = pd.to_datetime(
            data["Date"], errors="coerce"
        )

        data["Storage_Used_GB"] = pd.to_numeric(
            data["Storage_Used_GB"], errors="coerce"
        )

        data = data.dropna(
            subset=["Date", "Storage_Used_GB"]
        )

        data = data.sort_values("Date").reset_index(drop=True)

        if len(data) < 10:
            st.error("Please upload at least 10 valid records.")
            st.stop()

        if (data["Storage_Used_GB"] < 0).any():
            st.error("Storage usage cannot be negative.")
            st.stop()

        st.success("CSV file uploaded successfully!")

    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

else:
    data = sample_data.copy()
    st.info(
        "Showing sample simulated data. "
        "Upload your own CSV to analyze your data."
    )

# Storage overview
st.subheader("1. Storage Usage Overview")

current_usage = data["Storage_Used_GB"].iloc[-1]
available = capacity - current_usage
percentage = (current_usage / capacity) * 100

c1, c2, c3 = st.columns(3)

c1.metric("Current Storage", f"{current_usage:.1f} GB")
c2.metric("Available Storage", f"{available:.1f} GB")
c3.metric("Capacity Used", f"{percentage:.1f}%")

if current_usage >= capacity:
    st.error("⚠️ Storage capacity exceeded!")
elif current_usage >= capacity * threshold / 100:
    st.warning("⚠️ Storage usage has reached the warning threshold!")
else:
    st.success("Storage usage is below the warning threshold.")

# Historical graph
st.subheader("2. Historical Storage Usage")

fig, ax = plt.subplots()

ax.plot(
    data["Date"],
    data["Storage_Used_GB"],
    marker="o"
)

ax.set_xlabel("Date")
ax.set_ylabel("Storage Used (GB)")
ax.set_title("Historical Storage Consumption")
ax.grid(True)

plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Prepare machine learning data
X = np.arange(len(data)).reshape(-1, 1)
y = data["Storage_Used_GB"].values

# Chronological 80:20 train-test split
split_index = int(len(data) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

st.subheader("3. Machine Learning Model Comparison")

st.write(
    f"Training records: {len(X_train)} | "
    f"Testing records: {len(X_test)}"
)

st.info(
    "The first 80% of records are used for training, "
    "and the remaining 20% are used for testing."
)

# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = {}
test_predictions = {}
future_predictions = {}

# Evaluate models on unseen test data
for name, model in models.items():

    # Train using training data only
    model.fit(X_train, y_train)

    # Predict testing data
    test_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, test_pred)
    )

    results[name] = {
        "Test MAE": mae,
        "Test RMSE": rmse
    }

    test_predictions[name] = test_pred

# Compare test errors
results_df = pd.DataFrame(results).T

st.dataframe(
    results_df.style.format("{:.2f}"),
    width="stretch"
)

# Select best model using test RMSE
best_model_name = results_df["Test RMSE"].idxmin()

st.success(
    f"Best model based on test RMSE: {best_model_name}"
)

# Show best model's test predictions
st.subheader("4. Test Data Prediction")

best_test_predictions = test_predictions[best_model_name]

comparison_df = pd.DataFrame({
    "Date": data["Date"].iloc[split_index:].values,
    "Actual Storage (GB)": y_test,
    "Predicted Storage (GB)": best_test_predictions
})

st.dataframe(
    comparison_df.style.format({
        "Actual Storage (GB)": "{:.2f}",
        "Predicted Storage (GB)": "{:.2f}"
    }),
    width="stretch"
)

# Future forecasting
st.subheader(f"5. Future Storage Prediction ({days} Days)")

# Retrain each model using all available historical data
# before generating future forecasts.
future_X = np.arange(
    len(data), len(data) + days
).reshape(-1, 1)

for name, model in models.items():

    model.fit(X, y)

    future_predictions[name] = model.predict(future_X)

# Forecast using the best model selected during testing
forecast = future_predictions[best_model_name]

future_dates = pd.date_range(
    start=data["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=days
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Storage_GB": forecast
})

st.dataframe(
    forecast_df.style.format({
        "Predicted_Storage_GB": "{:.2f}"
    }),
    width="stretch"
)

# Forecast graph
fig2, ax2 = plt.subplots()

ax2.plot(
    data["Date"],
    y,
    label="Historical Usage",
    marker="o"
)

ax2.plot(
    future_dates,
    forecast,
    label="Predicted Usage",
    marker="o"
)

ax2.axhline(
    capacity,
    linestyle="--",
    label="Total Capacity"
)

ax2.axhline(
    capacity * threshold / 100,
    linestyle=":",
    label="Warning Threshold"
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Storage (GB)")
ax2.set_title("Storage Usage Forecast")
ax2.legend()
ax2.grid(True)

plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig2)
plt.close(fig2)

# Capacity warning
max_prediction = np.max(forecast)

if max_prediction >= capacity:
    st.error("⚠️ Prediction: Storage capacity may be exceeded.")

elif max_prediction >= capacity * threshold / 100:
    st.warning(
        "⚠️ Prediction: Storage may reach the warning threshold."
    )

else:
    st.success(
        "Predicted storage usage is within the safe capacity."
    )

# Project summary
st.subheader("6. Project Summary")

st.write(f"- Dataset records: {len(data)}")
st.write(f"- Training records: {len(X_train)}")
st.write(f"- Testing records: {len(X_test)}")
st.write(f"- Forecast period: {days} days")
st.write(f"- Selected model: {best_model_name}")
st.write(f"- Highest predicted usage: {max_prediction:.2f} GB")
st.write(f"- Total storage capacity: {capacity} GB")

st.caption(
    "CloudPredict | Academic Mini Project | Data Science"
)
