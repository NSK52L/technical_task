import pandas as pd
import joblib
from itertools import product

import config


# ============================================================
# DYNAMIC CANDIDATE EVALUATOR
# ============================================================
# This module evaluates possible operating configurations for
# the current simulated system conditions.
#
# It does NOT run the simulation or change physical devices.
# It only predicts, checks constraints, and recommends a choice.
#
# Assumption: both the Raspberry Pi and cloud computer remain
# powered on. The trained power model includes both idle loads.
# ============================================================


# Candidate configuration space: 2 x 2 x 4 x 2 = 32 choices.
EXECUTION_LOCATIONS = ["Edge", "Cloud"]
PROCESSING_MODES = ["Normal", "Throttled"]
CAMERA_RATES = [1, 2, 5, 10]
LAMP_STATES = [True, False]

# The decision is planned once per minute.
DECISION_INTERVAL_SECONDS = 60


# ============================================================
# 1. LOAD TRAINED MODELS
# ============================================================

def load_models():
    """Load the trained power and detection models."""

    try:
        power_model = joblib.load("power_model.pkl")
        detection_model = joblib.load("detection_model.pkl")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "A trained model file is missing. Run models.py first "
            "to generate power_model.pkl and detection_model.pkl."
        ) from error

    # Avoid spawning parallel workers for tiny candidate batches. On laptops,
    # n_jobs=-1 can add substantial overhead compared with a single worker.
    for model in (power_model, detection_model):
        try:
            params = model.get_params(deep=True)
            parallel_params = {key: 1 for key in params if key.endswith("n_jobs")}
            if parallel_params:
                model.set_params(**parallel_params)
        except (AttributeError, ValueError):
            pass

    return power_model, detection_model


# ============================================================
# 2. ENGINEERING CAPACITY MODEL
# ============================================================

def get_processing_capacity(location, mode):
    """Return the configured processing capacity in images/min."""

    if location == "Edge":
        if mode == "Normal":
            return config.cv_capacity_edge
        return config.cv_throttled_capacity_edge

    if location == "Cloud":
        if mode == "Normal":
            return config.cv_capacity_cloud
        return config.cv_throttled_capacity_cloud

    raise ValueError(f"Unknown execution location: {location}")


# ============================================================
# 3. VALIDATE CURRENT SYSTEM CONDITIONS
# ============================================================

def validate_conditions(storage_temperature, ambient_light):
    if not isinstance(storage_temperature, (int, float)):
        raise TypeError("storage_temperature must be numeric.")

    if not isinstance(ambient_light, (int, float)):
        raise TypeError("ambient_light must be numeric.")

    if ambient_light < 0:
        raise ValueError("ambient_light cannot be negative.")


# ============================================================
# 4. EVALUATE ALL CANDIDATE CONFIGURATIONS
# ============================================================

def evaluate_configurations(
    storage_temperature,
    ambient_light,
    power_model,
    detection_model
):
    """Evaluate all 32 configurations using batched ML predictions."""

    validate_conditions(storage_temperature, ambient_light)

    fan_on = storage_temperature >= config.fan_on_threshold
    candidates = list(product(
        EXECUTION_LOCATIONS,
        PROCESSING_MODES,
        CAMERA_RATES,
        LAMP_STATES
    ))

    # Prepare the complete candidate input matrices, then call each model
    # once rather than 32 times. This significantly reduces overhead.
    power_inputs = pd.DataFrame([{
        "execution_location": str(location),
        "processing_mode": str(mode),
        "camera_rate_images_per_min": camera_rate,
        "lamp_on": str(lamp_on),
        "storage_temperature_C": storage_temperature,
        "fan_on": str(fan_on)
    } for location, mode, camera_rate, lamp_on in candidates])

    detection_inputs = pd.DataFrame([{
        "execution_location": str(location),
        "processing_mode": str(mode),
        "camera_rate_images_per_min": camera_rate,
        "lamp_on": str(lamp_on),
        "ambient_light_lux": ambient_light
    } for location, mode, camera_rate, lamp_on in candidates])

    predicted_powers = power_model.predict(power_inputs)
    predicted_success_rates = detection_model.predict(detection_inputs)

    results = []
    for i, (location, mode, camera_rate, lamp_on) in enumerate(candidates):
        processing_capacity = get_processing_capacity(location, mode)
        processed_rate = min(camera_rate, processing_capacity)
        processing_ok = processing_capacity >= camera_rate

        predicted_power = float(predicted_powers[i])
        predicted_energy_wh = predicted_power * DECISION_INTERVAL_SECONDS / 3600
        power_ok = predicted_power <= config.max_power_w

        predicted_success_rate = float(max(0.0, min(1.0, predicted_success_rates[i])))
        expected_detections_per_min = processed_rate * predicted_success_rate
        detection_ok = expected_detections_per_min >= config.min_detections_per_minute
        temperature_ok = storage_temperature <= config.storage_temp_max

        feasible = processing_ok and detection_ok and power_ok and temperature_ok
        rejection_reasons = []
        if not processing_ok:
            rejection_reasons.append("insufficient_processing_capacity")
        if not detection_ok:
            rejection_reasons.append("insufficient_expected_detections")
        if not power_ok:
            rejection_reasons.append("power_budget_exceeded")
        if not temperature_ok:
            rejection_reasons.append("storage_temperature_exceeded")

        results.append({
            "location": location,
            "mode": mode,
            "camera_rate": camera_rate,
            "lamp_on": lamp_on,
            "storage_temperature_C": storage_temperature,
            "ambient_light_lux": ambient_light,
            "fan_on": fan_on,
            "processing_capacity_images_per_min": processing_capacity,
            "processed_rate_images_per_min": processed_rate,
            "predicted_power_W": predicted_power,
            "predicted_energy_Wh_per_decision": predicted_energy_wh,
            "predicted_detection_success_rate": predicted_success_rate,
            "expected_detections_per_min": expected_detections_per_min,
            "processing_ok": processing_ok,
            "detection_ok": detection_ok,
            "power_ok": power_ok,
            "temperature_ok": temperature_ok,
            "feasible": feasible,
            "rejection_reasons": ";".join(rejection_reasons)
        })

    return pd.DataFrame(results)


# ============================================================
# 5. SELECT THE BEST FEASIBLE CONFIGURATION
# ============================================================

def select_configuration(results):
    """
    Select the feasible candidate with the lowest predicted
    power. All candidates use the same 60-second decision
    interval, so minimizing predicted energy gives the same
    ranking.
    """

    feasible_results = results[results["feasible"]].copy()

    if feasible_results.empty:
        return None

    best = feasible_results.sort_values(
        by=["predicted_energy_Wh_per_decision", "predicted_power_W"]
    ).iloc[0]

    return best


# ============================================================
# 6. CONVENIENCE FUNCTION FOR ONE ORCHESTRATION DECISION
# ============================================================

def make_decision(
    storage_temperature,
    ambient_light,
    power_model,
    detection_model
):
    """Evaluate candidates and return the results and best choice."""

    results = evaluate_configurations(
        storage_temperature=storage_temperature,
        ambient_light=ambient_light,
        power_model=power_model,
        detection_model=detection_model
    )

    best = select_configuration(results)

    return results, best


# ============================================================
# 7. STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    power_model, detection_model = load_models()

    # Test conditions. These will later be supplied by the
    # dynamic simulation at each orchestration decision.
    test_temperature = 35.0
    test_ambient_light = 500.0

    results, best = make_decision(
        storage_temperature=test_temperature,
        ambient_light=test_ambient_light,
        power_model=power_model,
        detection_model=detection_model
    )

    print("\n========== CANDIDATE EVALUATOR ==========")
    print("Storage temperature:", test_temperature, "°C")
    print("Ambient light:", test_ambient_light, "lux")
    print("Configurations evaluated:", len(results))
    print("Feasible configurations:", int(results["feasible"].sum()))
    print("Infeasible configurations:", int((~results["feasible"]).sum()))

    print("\n========== FEASIBLE CANDIDATES ==========")

    feasible = results[results["feasible"]].sort_values(
        "predicted_power_W"
    )

    if feasible.empty:
        print("No feasible configuration found.")
    else:
        columns_to_show = [
            "location",
            "mode",
            "camera_rate",
            "lamp_on",
            "processing_capacity_images_per_min",
            "predicted_power_W",
            "predicted_detection_success_rate",
            "expected_detections_per_min"
        ]

        print(
            feasible[columns_to_show].to_string(
                index=False,
                formatters={
                    "predicted_power_W": "{:.2f}".format,
                    "predicted_detection_success_rate": "{:.2%}".format,
                    "expected_detections_per_min": "{:.2f}".format
                }
            )
        )

        print("\n========== SELECTED CONFIGURATION ==========")
        print("Execution location:", best["location"])
        print("Processing mode:", best["mode"])
        print("Camera rate:", best["camera_rate"], "images/min")
        print("Lamp ON:", best["lamp_on"])
        print(
            "Predicted power:",
            round(best["predicted_power_W"], 2),
            "W"
        )
        print(
            "Expected detections:",
            round(best["expected_detections_per_min"], 2),
            "per minute"
        )

    results.to_csv("candidate_evaluation.csv", index=False)
    print("\nFull evaluation saved to candidate_evaluation.csv")
