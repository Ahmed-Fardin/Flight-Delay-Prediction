import pandas as pd
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "Models" / "airintel_logistic_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "Models" / "airintel_preprocessor.joblib"

AIRLINE_HISTORY_PATH = PROJECT_ROOT / "Models" / "airline_history.csv"
ORIGIN_HISTORY_PATH = PROJECT_ROOT / "Models" / "origin_history.csv"
DESTINATION_HISTORY_PATH = PROJECT_ROOT / "Models" / "destination_history.csv"
ROUTE_HISTORY_PATH = PROJECT_ROOT / "Models" / "route_history.csv"


def load_resources():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    airline_history = pd.read_csv(AIRLINE_HISTORY_PATH)
    origin_history = pd.read_csv(ORIGIN_HISTORY_PATH)
    destination_history = pd.read_csv(DESTINATION_HISTORY_PATH)
    route_history = pd.read_csv(ROUTE_HISTORY_PATH)

    return (
        model,
        preprocessor,
        airline_history,
        origin_history,
        destination_history,
        route_history,
    )


def predict_flights(df):
    (
        model,
        preprocessor,
        airline_history,
        origin_history,
        destination_history,
        route_history,
    ) = load_resources()

    df = df.copy()

    scheduled = (
        df["SCHEDULED_DEPARTURE"]
        .astype(str)
        .str.zfill(4)
    )

    df["DEP_HOUR"] = scheduled.str[:2].astype(int)
    df["DEP_MINUTE"] = scheduled.str[2:].astype(int)

    df = df.merge(airline_history, on="AIRLINE", how="left")
    df = df.merge(origin_history, on="ORIGIN_AIRPORT", how="left")
    df = df.merge(
        destination_history,
        on="DESTINATION_AIRPORT",
        how="left"
    )
    df = df.merge(
        route_history,
        on=["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"],
        how="left"
    )

    global_delay_rate = airline_history["delay_rate"].mean()

    df["delay_rate"] = df["delay_rate"].fillna(global_delay_rate)
    df["origin_delay_rate"] = df["origin_delay_rate"].fillna(global_delay_rate)
    df["destination_delay_rate"] = df["destination_delay_rate"].fillna(global_delay_rate)
    df["route_delay_rate"] = df["route_delay_rate"].fillna(global_delay_rate)

    df["AIRLINE_DEP_HOUR"] = (
        df["AIRLINE"].astype(str)
        + "_"
        + df["DEP_HOUR"].astype(str)
    )

    feature_columns = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "MONTH",
        "DAY_OF_WEEK",
        "DEP_HOUR",
        "AIRLINE_DEP_HOUR",
        "SCHEDULED_TIME",
        "DISTANCE",
        "delay_rate",
        "origin_delay_rate",
        "destination_delay_rate",
        "route_delay_rate",
    ]

    X_new = df[feature_columns]

    # Apply the exact preprocessing used during training
    X_new_processed = preprocessor.transform(X_new)

    df["DELAY_PROBABILITY"] = model.predict_proba(
        X_new_processed
    )[:, 1]

    df["PREDICTED_DELAYED"] = (
        df["DELAY_PROBABILITY"] >= 0.25
    ).astype(int)

    return df