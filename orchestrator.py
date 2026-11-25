
import pandas as pd
import joblib
from itertools import product

import config


# ==================================================
# 1. LOAD TRAINED MODELS
# ==================================================

power_model = joblib.load("power_model.pkl")
detection_model = joblib.load("detection_model.pkl")


# ==================================================
# 2. CURRENT SYSTEM CONDITIONS
# ==================================================

storage_temperature = 35.0
ambient_light = 200.0


# ==================================================
# 3. POSSIBLE CONFIGURATIONS
# ==================================================

execution_locations = ["Edge", "Cloud"]
processing_modes = ["Normal", "Throttled"]
camera_rates = [1, 2, 5, 10]
lamp_states = [True, False]


# ==================================================
# 4. PROCESSING CAPACITY
# ==================================================

def get_processing_capacity(location, mode):

    if location == "Edge":

        if mode == "Normal":
            return config.cv_capacity_edge

        return config.cv_throttled_capacity_edge

    else:

        if mode == "Normal":
            return config.cv_capacity_cloud

        return config.cv_throttled_capacity_cloud


# ==================================================
# 5. EVALUATE CONFIGURATIONS
# ==================================================

def evaluate_configurations():

    results = []

    fan_on = storage_temperature >= config.fan_on_threshold

    for location, mode, rate, lamp_on in product(
        execution_locations,
        processing_modes,
        camera_rates,
        lamp_states
    ):

        # ------------------------------------------
        # A. Calculate processing capacity
        # ------------------------------------------

        processing_capacity = get_processing_capacity(
            location,
            mode
        )

        processed_rate = min(rate, processing_capacity)

        processing_ok = (
            processing_capacity >= rate
        )

        # ------------------------------------------
        # B. Predict total power
        # ------------------------------------------

        power_input = pd.DataFrame([{
            "execution_location": str(location),
            "processing_mode": str(mode),
            "camera_rate_images_per_min": rate,
            "lamp_on": str(lamp_on),
            "storage_temperature_C": storage_temperature,
            "fan_on": str(fan_on)
        }])

        predicted_power = power_model.predict(
            power_input
        )[0]

        power_ok = (
            predicted_power <= config.max_power_w
        )

        # ------------------------------------------
        # C. Predict detection success probability
        # ------------------------------------------

        detection_input = pd.DataFrame([{
            "execution_location": str(location),
            "processing_mode": str(mode),
            "camera_rate_images_per_min": rate,
            "lamp_on": str(lamp_on),
            "ambient_light_lux": ambient_light
        }])

        predicted_success_rate = detection_model.predict(
            detection_input
        )[0]

        # Keep probability within valid limits
        predicted_success_rate = max(
            0.0,
            min(1.0, predicted_success_rate)
        )

        expected_detections = (
            processed_rate * predicted_success_rate
        )

        detection_ok = (
            expected_detections >=
            config.min_detections_per_minute
        )

        # ------------------------------------------
        # D. Check temperature constraint
        # ------------------------------------------

        temperature_ok = (
            storage_temperature <= config.storage_temp_max
        )

        # ------------------------------------------
        # E. Overall feasibility
        # ------------------------------------------

        feasible = (
            processing_ok
            and detection_ok
            and power_ok
            and temperature_ok
        )

        # ------------------------------------------
        # F. Store results
        # ------------------------------------------

        results.append({

            "location": location,
            "mode": mode,
            "camera_rate": rate,
            "lamp_on": lamp_on,

            "processing_capacity": processing_capacity,
            "processed_rate": processed_rate,

            "predicted_power_W": predicted_power,

            "predicted_detection_success_rate":
                predicted_success_rate,

            "expected_detections_per_min":
                expected_detections,

            "processing_ok": processing_ok,
            "detection_ok": detection_ok,
            "power_ok": power_ok,
            "temperature_ok": temperature_ok,

            "feasible": feasible
        })

    return pd.DataFrame(results)


# ==================================================
# 6. SELECT BEST CONFIGURATION
# ==================================================

def select_configuration(results):

    feasible_results = results[
        results["feasible"] == True
    ]

    if feasible_results.empty:

        print("\nNo feasible configuration found.")

        return None

    # Minimize predicted power consumption
    best = feasible_results.sort_values(
        by="predicted_power_W"
    ).iloc[0]

    return best


# ==================================================
# 7. MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    print("========== ORCHESTRATOR ==========")

    print("\nCurrent system conditions:")
    print("Storage temperature:", storage_temperature, "°C")
    print("Ambient light:", ambient_light, "lux")

    results = evaluate_configurations()

    print(
        "\nTotal configurations evaluated:",
        len(results)
    )

    feasible_count = results["feasible"].sum()

    print("Feasible configurations:", feasible_count)

    print(
        "Infeasible configurations:",
        len(results) - feasible_count
    )

    # ------------------------------------------
    # Display feasible configurations
    # ------------------------------------------

    print("\n========== FEASIBLE CONFIGURATIONS ==========")

    feasible_results = results[
        results["feasible"] == True
    ]

    if not feasible_results.empty:

        print(
            feasible_results.sort_values(
                "predicted_power_W"
            ).to_string(
                index=False,
                formatters={
                    "predicted_power_W": "{:.2f}".format,
                    "predicted_detection_success_rate": "{:.2%}".format,
                    "expected_detections_per_min": "{:.2f}".format
                }
            )
        )

    # ------------------------------------------
    # Save evaluation results
    # ------------------------------------------

    results.to_csv(
        "orchestrator_evaluation.csv",
        index=False
    )

    print("\nEvaluation saved to orchestrator_evaluation.csv")

    # ------------------------------------------
    # Select best configuration
    # ------------------------------------------

    best = select_configuration(results)

    if best is not None:

        print("\n========== SELECTED CONFIGURATION ==========")

        print("Execution location:", best["location"])
        print("Processing mode:", best["mode"])
        print("Camera rate:", best["camera_rate"], "images/min")
        print("Lamp ON:", best["lamp_on"])

        print(
            "Processing capacity:",
            best["processing_capacity"],
            "images/min"
        )

        print(
            "Predicted power:",
            round(best["predicted_power_W"], 2),
            "W"
        )

        print(
            "Predicted detection success:",
            round(
                best["predicted_detection_success_rate"] * 100,
                2
            ),
            "%"
        )

        print(
            "Expected detections:",
            round(best["expected_detections_per_min"], 2),
            "per minute"
        )
