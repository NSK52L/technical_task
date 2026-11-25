
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error
)

import config


# ==================================================
# 1. LOAD DATASET
# ==================================================

data = pd.read_csv("training_data.csv")

print("Dataset shape:", data.shape)


# ==================================================
# 2. DEFINE FEATURES AND TARGETS
# ==================================================

power_features = [
    "execution_location",
    "processing_mode",
    "camera_rate_images_per_min",
    "lamp_on",
    "storage_temperature_C",
    "fan_on"
]

performance_features = [
    "execution_location",
    "processing_mode",
    "camera_rate_images_per_min"
]

detection_features = [
    "execution_location",
    "processing_mode",
    "camera_rate_images_per_min",
    "lamp_on",
    "ambient_light_lux"
]

y_power = data["total_power_W"]
y_performance = data["processed_rate_images_per_min"]
y_detection = data["observed_detection_success_rate"]


# ==================================================
# 3. PREPARE CATEGORICAL FEATURES
# ==================================================

def prepare_features(features):
    X = data[features].copy()

    for column in X.columns:
        if column in [
            "execution_location",
            "processing_mode",
            "lamp_on",
            "fan_on"
        ]:
            X[column] = X[column].astype(str)

    return X


X_power = prepare_features(power_features)
X_performance = prepare_features(performance_features)
X_detection = prepare_features(detection_features)


# ==================================================
# 4. GROUPED TRAIN / TEST SPLIT
# ==================================================

# All repetitions of the same configuration
# belong to the same group.

groups = (
    data[
        [
            "execution_location",
            "processing_mode",
            "camera_rate_images_per_min",
            "lamp_on"
        ]
    ]
    .astype(str)
    .agg("_".join, axis=1)
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(data, y_power, groups=groups)
)


# ==================================================
# 5. CREATE MODEL PIPELINES
# ==================================================


def create_pipeline(features, model):

    # Explicitly identify categorical columns
    categorical_columns = [
        "execution_location",
        "processing_mode",
        "lamp_on",
        "fan_on"
    ]

    categorical = [
        col for col in features.columns
        if col in categorical_columns
    ]

    numerical = [
        col for col in features.columns
        if col not in categorical
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical
            ),
            (
                "numerical",
                "passthrough",
                numerical
            )
        ]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])



def create_random_forest(features):

    return create_pipeline(
        features,
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1
        )
    )


def create_baseline(features):

    return create_pipeline(
        features,
        DummyRegressor(strategy="mean")
    )


# ==================================================
# 6. EVALUATION FUNCTION
# ==================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    median_error = median_absolute_error(y_test, predictions)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE:                   {mae:.4f}")
    print(f"RMSE:                  {rmse:.4f}")
    print(f"R²:                    {r2:.4f}")
    print(f"Median absolute error: {median_error:.4f}")

    return mae


# ==================================================
# 7. TRAIN AND EVALUATE EACH MODEL
# ==================================================

def train_and_evaluate(name, X, y, filename):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    model = create_random_forest(X)

    baseline = create_baseline(X)

    # Five-fold grouped cross-validation
    cv = GroupKFold(n_splits=5)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    print(f"\n========== {name} ==========")

    print(
        "Cross-validation MAE:",
        round(-cv_scores.mean(), 4),
        "+/-",
        round(cv_scores.std(), 4)
    )

    # Train baseline
    baseline.fit(X_train, y_train)

    # Train Random Forest
    model.fit(X_train, y_train)

    baseline_mae = evaluate_model(
        "DUMMY BASELINE",
        baseline,
        X_test,
        y_test
    )

    model_mae = evaluate_model(
        "RANDOM FOREST",
        model,
        X_test,
        y_test
    )

    improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100
        if baseline_mae > 0 else 0
    )

    print(f"\nMAE improvement over baseline: {improvement:.2f}%")

    joblib.dump(model, filename)

    print("Saved:", filename)

    return model


# ==================================================
# 8. TRAIN THE THREE MODELS
# ==================================================

power_model = train_and_evaluate(
    "POWER MODEL",
    X_power,
    y_power,
    "power_model.pkl"
)

performance_model = train_and_evaluate(
    "PERFORMANCE MODEL",
    X_performance,
    y_performance,
    "performance_model.pkl"
)

detection_model = train_and_evaluate(
    "DETECTION MODEL",
    X_detection,
    y_detection,
    "detection_model.pkl"
)


# ==================================================
# 9. TEST A CONFIGURATION
# ==================================================

def predict_configuration(
    location,
    mode,
    camera_rate,
    lamp_on,
    temperature,
    ambient_light
):

    fan_on = temperature >= config.fan_on_threshold

    power_input = pd.DataFrame([{
        "execution_location": str(location),
        "processing_mode": str(mode),
        "camera_rate_images_per_min": camera_rate,
        "lamp_on": str(lamp_on),
        "storage_temperature_C": temperature,
        "fan_on": str(fan_on)
    }])

    performance_input = pd.DataFrame([{
        "execution_location": str(location),
        "processing_mode": str(mode),
        "camera_rate_images_per_min": camera_rate
    }])

    detection_input = pd.DataFrame([{
        "execution_location": str(location),
        "processing_mode": str(mode),
        "camera_rate_images_per_min": camera_rate,
        "lamp_on": str(lamp_on),
        "ambient_light_lux": ambient_light
    }])

    predicted_power = power_model.predict(power_input)[0]

    predicted_performance = (
        performance_model.predict(performance_input)[0]
    )

    predicted_detection = (
        detection_model.predict(detection_input)[0]
    )

    return (
        predicted_power,
        predicted_performance,
        predicted_detection
    )


print("\n========== TEST CONFIGURATION ==========")

power, performance, detection = predict_configuration(
    location="Edge",
    mode="Normal",
    camera_rate=5,
    lamp_on=True,
    temperature=35,
    ambient_light=200
)

print(f"Predicted power: {power:.2f} W")
print(f"Predicted processing rate: {performance:.2f} images/min")
print(f"Predicted detection success rate: {detection:.2%}")
