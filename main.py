
from pathlib import Path
from datetime import timedelta
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "electricity_demand_model.pkl"
CALIBRATION_PATH = BASE_DIR / "margin_calibration.json"
DATA_PATH = BASE_DIR / "data" / "household_power_consumption.txt"

# =========================================================
# 2. LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
FEATURES = bundle["feature_names"]

PEAK_THRESHOLD = float(bundle["peak_threshold"])
PEAK_CUTOFF = float(bundle["peak_forecast_cutoff"])

MODEL_NAME = bundle.get("model_name", "Unknown Model")

if not CALIBRATION_PATH.exists():
    raise FileNotFoundError(
        f"Calibration file not found: {CALIBRATION_PATH}"
    )

with open(CALIBRATION_PATH, "r") as f:
    calibration = json.load(f)

# =========================================================
# 3. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="GridWise Electricity Forecasting API",
    description=(
        "Electricity demand forecasting, dynamic safety margin, "
        "peak demand alerts, and energy risk analysis."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 4. LOAD HISTORICAL DATA
# =========================================================

def load_historical_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    df = df.dropna(subset=["datetime", "Global_active_power"])

    hourly = (
        df.set_index("datetime")["Global_active_power"]
        .resample("h")
        .mean()
        .dropna()
    )

    if hourly.empty:
        raise ValueError("No valid hourly historical data found.")

    return hourly


history = load_historical_data()

# =========================================================
# 5. SAFETY MARGIN CALIBRATION
# =========================================================

def get_base_safety_margin(prediction):

    bands = calibration.get("bands", [])

    for band in bands:

        lower = float(band["lower"])
        upper = float(band["upper"])

        if lower <= prediction <= upper:
            return float(band["margin_95"])

    return float(calibration.get("global_margin", 0.0))


# =========================================================
# 6. DYNAMIC SAFETY MARGIN
# =========================================================

def get_dynamic_safety_margin(prediction, history_series):

    # Original calibrated margin
    base_margin = get_base_safety_margin(prediction)

    # Recent demand volatility (last 24 hours)
    recent_values = history_series.iloc[-24:].astype(float)

    if len(recent_values) > 1:
        recent_std = float(recent_values.std())
    else:
        recent_std = 0.0

    # Small additional reserve based on recent demand variability
    volatility_adjustment = recent_std * 0.10

    # Extra reserve for high-demand conditions
    demand_adjustment = 0.0

    if prediction >= PEAK_CUTOFF:
        demand_adjustment = prediction * 0.05

    elif prediction >= PEAK_THRESHOLD:
        demand_adjustment = prediction * 0.02

    dynamic_margin = (
        base_margin
        + volatility_adjustment
        + demand_adjustment
    )

    dynamic_margin = max(0.0, dynamic_margin)

    return {
        "base_margin": float(base_margin),
        "volatility_adjustment": float(volatility_adjustment),
        "demand_adjustment": float(demand_adjustment),
        "dynamic_margin": float(dynamic_margin),
        "recent_volatility": float(recent_std)
    }


# =========================================================
# 7. ENERGY RISK SCORE
# =========================================================

def calculate_energy_risk(prediction, history_series):

    recent_values = history_series.iloc[-24:].astype(float)

    if len(recent_values) > 1:

        recent_mean = float(recent_values.mean())
        recent_std = float(recent_values.std())

    else:

        recent_mean = float(prediction)
        recent_std = 0.0

    # -----------------------------------------------------
    # Component 1: Demand risk (maximum 60 points)
    # -----------------------------------------------------

    if PEAK_CUTOFF > 0:

        demand_ratio = prediction / PEAK_CUTOFF

        demand_score = min(
            max(demand_ratio, 0.0),
            1.0
        ) * 60

    else:

        demand_score = 0.0

    # -----------------------------------------------------
    # Component 2: Volatility risk (maximum 25 points)
    # -----------------------------------------------------

    if recent_mean > 0:

        volatility_ratio = recent_std / recent_mean

    else:

        volatility_ratio = 0.0

    volatility_score = min(
        max(volatility_ratio, 0.0),
        1.0
    ) * 25

    # -----------------------------------------------------
    # Component 3: Peak demand risk (maximum 15 points)
    # -----------------------------------------------------

    peak_score = 15 if prediction >= PEAK_CUTOFF else 0

    # -----------------------------------------------------
    # Total score
    # -----------------------------------------------------

    risk_score = round(
        min(
            demand_score + volatility_score + peak_score,
            100
        )
    )

    if risk_score <= 30:
        risk_level = "Low"

    elif risk_score <= 60:
        risk_level = "Moderate"

    elif risk_score <= 80:
        risk_level = "High"

    else:
        risk_level = "Critical"

    return {
        "risk_score": int(risk_score),
        "risk_level": risk_level,
        "demand_risk_points": round(float(demand_score), 2),
        "volatility_risk_points": round(float(volatility_score), 2),
        "peak_risk_points": int(peak_score),
        "volatility_ratio": round(float(volatility_ratio), 4)
    }
# =========================================================
# UNUSUAL CONSUMPTION DETECTION
# =========================================================

def detect_unusual_consumption(prediction, forecast_time, history_series):

    forecast_time = pd.Timestamp(forecast_time)

    # Get historical consumption for the same hour of the day
    same_hour_data = history_series[
        history_series.index.hour == forecast_time.hour
    ]

    # Use the most recent 30 matching hourly observations
    baseline_data = same_hour_data.iloc[-30:]

    if baseline_data.empty:

        normal_usage = float(history_series.mean())

    else:

        normal_usage = float(baseline_data.mean())

    # Calculate deviation from normal usage
    difference = prediction - normal_usage

    if normal_usage > 0:

        percentage_change = (difference / normal_usage) * 100

    else:

        percentage_change = 0.0

    # Determine consumption status
    if percentage_change < 15:

        status = "Normal"
        alert = False
        severity = "LOW"

    elif percentage_change < 30:

        status = "Slight Increase"
        alert = False
        severity = "MODERATE"

    elif percentage_change < 50:

        status = "High Consumption"
        alert = True
        severity = "HIGH"

    else:

        status = "Unusual Consumption"
        alert = True
        severity = "CRITICAL"

    return {
        "normal_hourly_consumption": round(normal_usage, 4),
        "predicted_consumption": round(float(prediction), 4),
        "difference": round(float(difference), 4),
        "percentage_change": round(float(percentage_change), 2),
        "consumption_status": status,
        "unusual_consumption_alert": alert,
        "severity": severity
    }

# =========================================================
# 8. FEATURE ENGINEERING
# =========================================================

def build_features(history_series, forecast_time):

    forecast_time = pd.Timestamp(forecast_time)

    values = history_series.astype(float)

    if len(values) < 8760:
        raise ValueError(
            "At least 8760 hourly historical values are required "
            "to calculate lag_8760."
        )

    lag_1 = float(values.iloc[-1])
    lag_24 = float(values.iloc[-24])
    lag_168 = float(values.iloc[-168])
    lag_8760 = float(values.iloc[-8760])

    rolling_mean_24 = float(values.iloc[-24:].mean())
    rolling_mean_168 = float(values.iloc[-168:].mean())

    hour = forecast_time.hour
    day_of_week = forecast_time.dayofweek
    month = forecast_time.month
    is_weekend = int(day_of_week >= 5)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    year_sin = np.sin(
        2 * np.pi * forecast_time.dayofyear / 365.25
    )

    year_cos = np.cos(
        2 * np.pi * forecast_time.dayofyear / 365.25
    )

    diff_1_24 = lag_1 - lag_24
    diff_24_168 = lag_24 - lag_168

    ratio_roll = rolling_mean_24 / (rolling_mean_168 + 1e-6)

    lag_mean = np.mean([
        lag_24,
        lag_168,
        lag_8760
    ])

    features = {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "lag_8760": lag_8760,
        "rolling_mean_24": rolling_mean_24,
        "rolling_mean_168": rolling_mean_168,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "year_sin": year_sin,
        "year_cos": year_cos,
        "diff_1_24": diff_1_24,
        "diff_24_168": diff_24_168,
        "ratio_roll": ratio_roll,
        "lag_mean": lag_mean
    }

    feature_df = pd.DataFrame([features])

    # Ensure exact model feature order
    feature_df = feature_df[FEATURES]

    return feature_df


# =========================================================
# 9. SINGLE-HOUR PREDICTION
# =========================================================

def predict_one(history_series, forecast_time):

    X = build_features(
        history_series,
        forecast_time
    )

    prediction = float(model.predict(X)[0])

    # Dynamic margin calculation
    margin_info = get_dynamic_safety_margin(
        prediction,
        history_series
    )

    margin = margin_info["dynamic_margin"]

    recommended_capacity = prediction + margin

    # Peak warning
    
# =========================================================
# PEAK-RISK NOTIFICATION
# =========================================================

    # =========================================================
    # PEAK-RISK NOTIFICATION
    # =========================================================

    peak_alert = prediction >= PEAK_CUTOFF

    if prediction >= PEAK_CUTOFF * 1.20:

        peak_alert_level = "CRITICAL"
        peak_message = "Very high demand expected. Reduce simultaneous use of heavy appliances."
        peak_recommendation = "Avoid operating multiple high-power appliances simultaneously."

    elif peak_alert:

        peak_alert_level = "HIGH"
        peak_message = "High electricity demand is expected."
        peak_recommendation = "Consider shifting heavy appliance usage to another hour."

    elif prediction >= PEAK_THRESHOLD:

        peak_alert_level = "MODERATE"
        peak_message = "Demand is approaching the peak demand range."
        peak_recommendation = "Monitor electricity usage and avoid unnecessary high-power loads."

    else:

        peak_alert_level = "NORMAL"
        peak_message = "Electricity demand is expected to remain within the normal range."
        peak_recommendation = "No immediate action required."

    # =========================================================
    # ENERGY RISK SCORE
    # =========================================================

    risk_info = calculate_energy_risk(
        prediction,
        history_series
    )

    # =========================================================
    # UNUSUAL CONSUMPTION DETECTION
    # =========================================================

    unusual_info = detect_unusual_consumption(
        prediction,
        forecast_time,
        history_series
    )

    # =========================================================
    # RETURN PREDICTION RESULTS
    # =========================================================

    return {
        "datetime": str(forecast_time),
        "predicted_demand": round(prediction, 4),

        "safety_margin": round(margin, 4),
        "base_safety_margin": round(
            margin_info["base_margin"], 4
        ),
        "volatility_adjustment": round(
            margin_info["volatility_adjustment"], 4
        ),
        "demand_adjustment": round(
            margin_info["demand_adjustment"], 4
        ),

        "recommended_capacity": round(
            recommended_capacity, 4
        ),

        "peak_threshold": round(
            PEAK_THRESHOLD, 4
        ),
        "peak_forecast_cutoff": round(
            PEAK_CUTOFF, 4
        ),

        "peak_alert": bool(peak_alert),
        "peak_alert_level": peak_alert_level,
        "peak_risk_message": peak_message,
        "peak_recommendation": peak_recommendation,

        "energy_risk_score": risk_info["risk_score"],
        "energy_risk_level": risk_info["risk_level"],

        "risk_breakdown": {
            "demand_risk_points": risk_info["demand_risk_points"],
            "volatility_risk_points": risk_info["volatility_risk_points"],
            "peak_risk_points": risk_info["peak_risk_points"]
        },

        "unusual_consumption": unusual_info,

        "model": MODEL_NAME
    }# =========================================================
# HOUSEHOLD CONSUMPTION COMPARISON
# =========================================================

def compare_consumption(current, previous):

    difference = current - previous

    if previous > 0:
        percentage_change = (difference / previous) * 100
    else:
        percentage_change = 0.0

    if percentage_change > 5:
        status = "Increased"

    elif percentage_change < -5:
        status = "Decreased"

    else:
        status = "Stable"

    return {
        "current_consumption": round(float(current), 4),
        "previous_consumption": round(float(previous), 4),
        "difference": round(float(difference), 4),
        "percentage_change": round(float(percentage_change), 2),
        "status": status
    }


def get_household_comparison(history_series):

    # Daily consumption
    daily_data = history_series.resample("D").sum()

    complete_days = daily_data[
        history_series.resample("D").count() >= 24
    ]

    # Weekly consumption
    weekly_data = history_series.resample("W").sum()

    complete_weeks = weekly_data[
        history_series.resample("W").count() >= 168
    ]

    # Daily comparison
    if len(complete_days) >= 2:

        daily_comparison = compare_consumption(
            complete_days.iloc[-1],
            complete_days.iloc[-2]
        )

    else:

        daily_comparison = {
            "status": "Insufficient Data"
        }

    # Weekly comparison
    if len(complete_weeks) >= 2:

        weekly_comparison = compare_consumption(
            complete_weeks.iloc[-1],
            complete_weeks.iloc[-2]
        )

    else:

        weekly_comparison = {
            "status": "Insufficient Data"
        }

    return {
        "daily_comparison": daily_comparison,
        "weekly_comparison": weekly_comparison
    }

# ==========================================================
# SMART CAPACITY PLANNER
# ==========================================================

def smart_capacity_planner(
    predicted_peak_kw: float,
    safety_margin_percent: float,
    available_capacity_kw: float = None
):
    """
    Analyze predicted electricity demand and recommend capacity.

    The ML forecast is not changed.
    """

    predicted_peak_kw = max(0.0, float(predicted_peak_kw))
    safety_margin_percent = max(0.0, float(safety_margin_percent))

    # Recommended capacity with safety margin
    recommended_capacity_kw = (
        predicted_peak_kw * (1 + safety_margin_percent / 100)
    )

    result = {
        "predicted_peak_kw": round(predicted_peak_kw, 3),
        "safety_margin_percent": round(safety_margin_percent, 2),
        "recommended_capacity_kw": round(recommended_capacity_kw, 3),
        "available_capacity_kw": available_capacity_kw,
        "capacity_gap_kw": None,
        "utilization_percent": None,
        "capacity_status": "capacity_not_provided"
    }

    # Optional capacity analysis
    if available_capacity_kw is not None:

        available_capacity_kw = float(available_capacity_kw)

        if available_capacity_kw <= 0:
            raise ValueError("Available capacity must be greater than zero.")

        # Capacity required beyond the current available capacity
        capacity_gap_kw = max(
            0.0,
            recommended_capacity_kw - available_capacity_kw
        )

        # Actual forecast demand compared to available capacity
        utilization_percent = (
            predicted_peak_kw / available_capacity_kw
        ) * 100

        if predicted_peak_kw > available_capacity_kw:
            capacity_status = "overloaded"

        elif recommended_capacity_kw > available_capacity_kw:
            capacity_status = "insufficient_safety_margin"

        elif utilization_percent >= 90:
            capacity_status = "near_limit"

        else:
            capacity_status = "sufficient"

        result.update({
            "available_capacity_kw": round(available_capacity_kw, 3),
            "capacity_gap_kw": round(capacity_gap_kw, 3),
            "utilization_percent": round(utilization_percent, 2),
            "capacity_status": capacity_status
        })

    return result
# =========================================================
# 10. ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "GridWise Electricity Forecasting API is running",
        "model": MODEL_NAME,
        "features": len(FEATURES),
        "enhancements": [
            "Dynamic Safety Margin",
            "Peak Demand Warning",
            "Energy Risk Score"
        ],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "forecast": "/forecast",
            "docs": "/docs"
        }
    }


# =========================================================
# 11. HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "dataset_loaded": not history.empty,
        "dataset_records": len(history),
        "model_name": MODEL_NAME,
        "feature_count": len(FEATURES)
    }


# =========================================================
# 12. NEXT-HOUR PREDICTION
# =========================================================

@app.get("/predict")
def predict():

    try:

        next_time = history.index[-1] + timedelta(hours=1)

        result = predict_one(
            history,
            next_time
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# 13. MULTI-HOUR FORECAST
# =========================================================

@app.get("/forecast")
def forecast(
    hours: int = Query(24, ge=1, le=168)
):

    try:

        simulated_history = history.copy()

        forecast_results = []

        next_time = simulated_history.index[-1] + timedelta(hours=1)

        for i in range(hours):

            forecast_time = next_time + timedelta(hours=i)

            result = predict_one(
                simulated_history,
                forecast_time
            )

            forecast_results.append(result)

            # Recursive forecasting
            simulated_history.loc[forecast_time] = (
                result["predicted_demand"]
            )

        predictions = [
            r["predicted_demand"]
            for r in forecast_results
        ]

        capacities = [
            r["recommended_capacity"]
            for r in forecast_results
        ]

        # Find peak forecast
        peak_index = int(np.argmax(predictions))

        peak_forecast = forecast_results[peak_index]

        # Peak warning information
        peak_warning = {
            "peak_demand": peak_forecast["predicted_demand"],
            "peak_datetime": peak_forecast["datetime"],
            "peak_alert": peak_forecast["peak_alert"],
            "peak_alert_level": peak_forecast["peak_alert_level"]
        }

        # Overall forecast risk
        risk_scores = [
            r["energy_risk_score"]
            for r in forecast_results
        ]

        maximum_risk = max(risk_scores)

        if maximum_risk <= 30:
            overall_risk_level = "Low"

        elif maximum_risk <= 60:
            overall_risk_level = "Moderate"

        elif maximum_risk <= 80:
            overall_risk_level = "High"

        else:
            overall_risk_level = "Critical"

        return {
            "model": MODEL_NAME,
            "forecast_hours": hours,

            "peak_threshold": PEAK_THRESHOLD,

            "maximum_forecast": round(
                max(predictions), 4
            ),

            "maximum_recommended_capacity": round(
                max(capacities), 4
            ),

            "peak_hours": sum(
                r["peak_alert"] for r in forecast_results
            ),

            "peak_warning": peak_warning,

            "maximum_energy_risk_score": maximum_risk,

            "overall_energy_risk_level": overall_risk_level,

            "forecasts": forecast_results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================================================
# HOUSEHOLD COMPARISON API
# =========================================================

@app.get("/comparison")
def household_comparison():

    comparison = get_household_comparison(history)

    return {
        "status": "success",
        "comparison": comparison
    }
# ==========================================================
# 14. SMART CAPACITY PLANNER
# ==========================================================

@app.get("/capacity-plan")
def capacity_plan(
    hours: int = Query(24, ge=1, le=168),
    available_capacity_kw: float = Query(None, gt=0)
):

    try:

        # --------------------------------------------------
        # 1. Generate forecast using the existing forecast API
        # --------------------------------------------------

        forecast_data = forecast(hours=hours)

        # IMPORTANT: The correct key is "forecasts"
        forecast_results = forecast_data.get("forecasts", [])

        if not forecast_results:

            raise HTTPException(
                status_code=500,
                detail="Forecast data is unavailable."
            )

        # --------------------------------------------------
        # 2. Find the highest predicted demand
        # --------------------------------------------------

        peak_result = max(
            forecast_results,
            key=lambda item: item["predicted_demand"]
        )

        predicted_peak_kw = float(
            peak_result["predicted_demand"]
        )

        # --------------------------------------------------
        # 3. Find the highest recommended capacity
        # --------------------------------------------------

        capacity_peak_result = max(
            forecast_results,
            key=lambda item: item["recommended_capacity"]
        )

        recommended_capacity_kw = float(
            capacity_peak_result["recommended_capacity"]
        )

        recommended_datetime = capacity_peak_result["datetime"]

        # --------------------------------------------------
        # 4. Calculate available capacity analysis
        # --------------------------------------------------

        capacity_gap_kw = None
        capacity_surplus_kw = None
        utilization_percent = None

        capacity_status = "capacity_not_provided"

        if available_capacity_kw is not None:

            available_capacity_kw = float(
                available_capacity_kw
            )

            # Capacity shortage
            capacity_gap_kw = max(
                0.0,
                recommended_capacity_kw - available_capacity_kw
            )

            # Capacity surplus
            capacity_surplus_kw = max(
                0.0,
                available_capacity_kw - recommended_capacity_kw
            )

            # Actual demand compared with available capacity
            utilization_percent = (
                predicted_peak_kw / available_capacity_kw
            ) * 100

            # --------------------------------------------------
            # 5. Determine capacity status
            # --------------------------------------------------

            if predicted_peak_kw > available_capacity_kw:

                capacity_status = "overloaded"

            elif recommended_capacity_kw > available_capacity_kw:

                capacity_status = "insufficient_safety_margin"

            elif utilization_percent >= 90:

                capacity_status = "near_limit"

            else:

                capacity_status = "sufficient"

        # --------------------------------------------------
        # 6. Return Smart Capacity Planner results
        # --------------------------------------------------

        return {

            "planner": "GridWise Smart Capacity Planner",

            "forecast_hours": hours,

            "predicted_peak_demand_kw": round(
                predicted_peak_kw, 4
            ),

            "peak_datetime": peak_result["datetime"],

            "recommended_capacity_kw": round(
                recommended_capacity_kw, 4
            ),

            "recommended_capacity_datetime": recommended_datetime,

            "dynamic_safety_margin_kw": round(
                float(capacity_peak_result["safety_margin"]), 4
            ),

            "available_capacity_kw": available_capacity_kw,

            "capacity_gap_kw": (
                round(capacity_gap_kw, 4)
                if capacity_gap_kw is not None
                else None
            ),

            "capacity_surplus_kw": (
                round(capacity_surplus_kw, 4)
                if capacity_surplus_kw is not None
                else None
            ),

            "utilization_percent": (
                round(utilization_percent, 2)
                if utilization_percent is not None
                else None
            ),

            "capacity_status": capacity_status,

            "peak_alert": peak_result["peak_alert"],

            "energy_risk_level": peak_result["energy_risk_level"]

        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
# ==========================================================
# 15. MODEL EVALUATION METRICS
# ==========================================================

@app.get("/metrics")
def get_model_metrics():

    try:
        metrics = bundle.get("test_metrics", {})

        if not metrics:
            raise HTTPException(
                status_code=404,
                detail="Model evaluation metrics not found."
            )

        r2_score = float(metrics.get("R2", 0))

        return {
            "model_name": bundle.get(
                "model_name",
                "GridWise Electricity Demand Forecast Model"
            ),
            "metrics": {
                "MAE": round(float(metrics.get("MAE", 0)), 4),
                "RMSE": round(float(metrics.get("RMSE", 0)), 4),
                "R2": round(r2_score, 4)
            },
            "performance": {
                "accuracy_score_percent": round(r2_score * 100, 2)
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 