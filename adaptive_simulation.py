import csv
from collections import deque
import numpy as np
import pandas as pd

import config
import environment
import devices
import candidate_evaluator as evaluator


# ============================================================
# PAIRED FIXED-BASELINE VS ADAPTIVE SIMULATION
# ============================================================
# Both simulations use the same environmental trace and random
# detection draws. The baseline configuration is fixed, while
# the adaptive system reevaluates every 60 seconds.
# ============================================================

DT = config.time_step_seconds
TOTAL_STEPS = int(config.sim_hours * 3600 / DT)
DECISION_INTERVAL = evaluator.DECISION_INTERVAL_SECONDS
WINDOW_STEPS = int(60 / DT)

if DT <= 0 or 60 % DT != 0:
    raise ValueError("The timestep must be positive and divide 60 seconds.")
if DECISION_INTERVAL % DT != 0:
    raise ValueError("Decision interval must be a multiple of timestep.")

POWER_MODEL, DETECTION_MODEL = evaluator.load_models()


def make_environment_trace():
    """Generate a shared environment and shared detection random draws."""
    env_rng = np.random.default_rng(config.random_seed)
    detection_rng = np.random.default_rng(config.random_seed + 1)

    hours = np.arange(TOTAL_STEPS) * DT / 3600
    solar = np.array([environment.solar_power(h % 24) for h in hours])
    ambient_temp = np.array([
        environment.ambient_temperature(h % 24, env_rng) for h in hours
    ])
    ambient_light = np.array([
        environment.ambient_light(h % 24, env_rng) for h in hours
    ])

    # Enough random draws for the maximum camera rate (10 images/min).
    # 10 images/min and a 10-second timestep means at most 2 images/step.
    detection_draws = detection_rng.random((TOTAL_STEPS, 4))

    return hours, solar, ambient_temp, ambient_light, detection_draws


def run_simulation(adaptive, trace):
    hours, solar_trace, temp_trace, light_trace, detection_draws = trace

    name = "adaptive" if adaptive else "baseline"
    output_csv = "paired_baseline_results.csv" if not adaptive else "adaptive_results.csv"
    decision_csv = "adaptation_log.csv" if adaptive else None

    # Fixed baseline settings
    current_location = "Edge"
    current_mode = "Normal"
    current_camera_rate = config.camera_rate_images_per_minute
    current_lamp = True

    storage_temp = config.storage_temp_initial
    fan_state = False

    capture_credit = 0.0
    processing_credit = 0.0
    queue_size = 0

    total_captured = 0
    total_processed = 0
    total_successful = 0
    total_energy_wh = 0.0
    max_total_power = 0.0
    max_grid_power = 0.0
    power_violations = 0
    failed_windows = 0
    violation_steps = 0
    in_violation_episode = False
    detection_window = deque(maxlen=WINDOW_STEPS)

    rows = []
    decisions = []

    # Precompute baseline detection predictions in ONE batch. The baseline
    # configuration is fixed; only ambient light changes over time.
    baseline_probabilities = None
    if not adaptive:
        baseline_inputs = pd.DataFrame({
            "execution_location": [current_location] * TOTAL_STEPS,
            "processing_mode": [current_mode] * TOTAL_STEPS,
            "camera_rate_images_per_min": [current_camera_rate] * TOTAL_STEPS,
            "lamp_on": [str(current_lamp)] * TOTAL_STEPS,
            "ambient_light_lux": light_trace
        })
        baseline_probabilities = np.clip(
            DETECTION_MODEL.predict(baseline_inputs), 0, 1
        )

    # Cache adaptive detection predictions. The same configuration and
    # rounded light level often recur, so repeated model calls are avoided.
    prediction_cache = {}

    for step in range(TOTAL_STEPS):
        time_s = step * DT
        hour = hours[step] % 24
        solar = solar_trace[step]
        ambient_temp = temp_trace[step]
        ambient_light = light_trace[step]

        # Adaptive controller makes a decision at t=0 and every minute.
        if adaptive and step % int(DECISION_INTERVAL / DT) == 0:
            evaluated, best = evaluator.make_decision(
                storage_temperature=storage_temp,
                ambient_light=ambient_light,
                power_model=POWER_MODEL,
                detection_model=DETECTION_MODEL
            )

            previous = (
                current_location, current_mode,
                current_camera_rate, current_lamp
            )

            if best is not None:
                current_location = str(best["location"])
                current_mode = str(best["mode"])
                current_camera_rate = int(best["camera_rate"])
                current_lamp = bool(best["lamp_on"])

                decision_reason = "lowest_predicted_energy_feasible"
            else:
                # Safety fallback: use a high-capacity configuration,
                # lamp ON, and maximum available camera rate.
                current_location = "Cloud"
                current_mode = "Normal"
                current_camera_rate = 10
                current_lamp = True
                decision_reason = "no_feasible_candidate_fallback"

            changed = previous != (
                current_location, current_mode,
                current_camera_rate, current_lamp
            )

            decisions.append({
                "time_seconds": time_s,
                "hour": hour,
                "ambient_light_lux": ambient_light,
                "storage_temperature_before_decision_C": storage_temp,
                "previous_location": previous[0],
                "previous_mode": previous[1],
                "previous_camera_rate": previous[2],
                "previous_lamp_on": previous[3],
                "selected_location": current_location,
                "selected_mode": current_mode,
                "selected_camera_rate": current_camera_rate,
                "selected_lamp_on": current_lamp,
                "configuration_changed": changed,
                "decision_reason": decision_reason,
                "feasible_candidates": int(evaluated["feasible"].sum()),
                "predicted_power_W": (
                    float(best["predicted_power_W"]) if best is not None else np.nan
                ),
                "predicted_detection_success_rate": (
                    float(best["predicted_detection_success_rate"])
                    if best is not None else np.nan
                ),
                "expected_detections_per_min": (
                    float(best["expected_detections_per_min"])
                    if best is not None else np.nan
                )
            })

        # Camera acquisition and immediate storage
        capture_credit += current_camera_rate * DT / 60
        images_captured = int(capture_credit)
        capture_credit -= images_captured
        total_captured += images_captured
        queue_size += images_captured

        # Engineering processing capacity
        capacity = evaluator.get_processing_capacity(
            current_location, current_mode
        )
        processing_credit += capacity * DT / 60
        images_processed = min(queue_size, int(processing_credit))
        queue_size -= images_processed
        processing_credit -= images_processed
        total_processed += images_processed

        # Actual detections use the trained detection model's estimate
        # under the current selected configuration and environment.
        if not adaptive:
            success_probability = float(baseline_probabilities[step])
        else:
            cache_key = (
                current_location, current_mode, current_camera_rate,
                current_lamp, round(float(ambient_light), 1)
            )
            if cache_key not in prediction_cache:
                detection_input = pd.DataFrame([{
                    "execution_location": current_location,
                    "processing_mode": current_mode,
                    "camera_rate_images_per_min": current_camera_rate,
                    "lamp_on": str(current_lamp),
                    "ambient_light_lux": ambient_light
                }])
                prediction_cache[cache_key] = float(np.clip(
                    DETECTION_MODEL.predict(detection_input)[0], 0, 1
                ))
            success_probability = prediction_cache[cache_key]

        draws = detection_draws[step, :images_processed]
        successful = int(np.sum(draws < success_probability))
        total_successful += successful

        # Rolling 60-second requirement. Only complete windows count.
        detection_window.append(successful)
        rolling_detections = sum(detection_window)
        full_window = len(detection_window) == WINDOW_STEPS
        window_failed = (
            full_window and
            rolling_detections < config.min_detections_per_minute
        )

        if window_failed:
            violation_steps += 1
            if not in_violation_episode:
                failed_windows += 1
                in_violation_episode = True
        else:
            in_violation_episode = False

        # Approximate CPU utilization based on selected processing load.
        if current_location == "Edge":
            edge_cpu = min(100.0, current_camera_rate / capacity * 100)
            cloud_cpu = 0.0
        else:
            edge_cpu = 0.0
            cloud_cpu = min(100.0, current_camera_rate / capacity * 100)

        edge_p = devices.edge_power(edge_cpu)
        cloud_p = devices.cloud_power(cloud_cpu)
        camera_p = devices.camera_power(True)
        lamp_p = devices.lamp_power(current_lamp)
        storage_p = devices.storage_power(images_captured, DT)

        fan_state = devices.update_fan_state(storage_temp, fan_state)
        fan_p = devices.fan_power(fan_state)
        sensor_p = devices.sensor_power()

        total_p = devices.total_power(
            edge_p, cloud_p, camera_p, lamp_p,
            storage_p, fan_p, sensor_p
        )
        grid_p = max(0.0, total_p - solar)

        max_total_power = max(max_total_power, total_p)
        max_grid_power = max(max_grid_power, grid_p)
        if total_p > config.max_power_w:
            power_violations += 1

        energy_this_step = total_p * DT / 3600
        total_energy_wh += energy_this_step

        storage_temp = devices.update_storage_temp(
            current_temp=storage_temp,
            ambient_temp=ambient_temp,
            storage_power_val=storage_p,
            fan_on=fan_state,
            time_step_seconds=DT
        )

        rows.append({
            "time_seconds": time_s,
            "hour": hour,
            "ambient_light_lux": ambient_light,
            "ambient_temperature_C": ambient_temp,
            "solar_power_W": solar,
            "execution_location": current_location,
            "processing_mode": current_mode,
            "camera_rate_images_per_min": current_camera_rate,
            "lamp_on": current_lamp,
            "processing_capacity_images_per_min": capacity,
            "captured_images": images_captured,
            "processed_images": images_processed,
            "successful_detections": successful,
            "predicted_detection_success_rate": success_probability,
            "queue_size": queue_size,
            "storage_temperature_C": storage_temp,
            "fan_on": fan_state,
            "edge_power_W": edge_p,
            "cloud_power_W": cloud_p,
            "camera_power_W": camera_p,
            "lamp_power_W": lamp_p,
            "storage_power_W": storage_p,
            "fan_power_W": fan_p,
            "sensor_power_W": sensor_p,
            "total_power_W": total_p,
            "grid_power_W": grid_p,
            "energy_Wh": total_energy_wh,
            "rolling_detections_60s": rolling_detections,
            "complete_detection_window": full_window,
            "detection_window_violation": window_failed
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)

    if adaptive:
        pd.DataFrame(decisions).to_csv(decision_csv, index=False)

    summary = {
        "system": name,
        "simulated_hours": config.sim_hours,
        "captured_images": total_captured,
        "processed_images": total_processed,
        "successful_detections": total_successful,
        "images_remaining_in_queue": queue_size,
        "total_energy_Wh": total_energy_wh,
        "total_energy_kWh": total_energy_wh / 1000,
        "max_total_power_W": max_total_power,
        "max_grid_power_W": max_grid_power,
        "power_budget_violation_steps": power_violations,
        "failed_60s_detection_windows": failed_windows,
        "violation_steps_in_failed_windows": violation_steps,
        "final_storage_temperature_C": storage_temp,
        "fan_on_steps": int(result["fan_on"].sum()),
        "lamp_on_steps": int(result["lamp_on"].sum()),
        "configuration_changes": (
            int(pd.DataFrame(decisions)["configuration_changed"].sum())
            if adaptive and decisions else 0
        )
    }

    return result, summary


def main():
    print("Generating shared environment trace...")
    trace = make_environment_trace()

    print("\nRunning fixed baseline...")
    baseline_df, baseline_summary = run_simulation(False, trace)

    print("\nRunning adaptive orchestration...")
    adaptive_df, adaptive_summary = run_simulation(True, trace)

    comparison = pd.DataFrame([baseline_summary, adaptive_summary])
    comparison.to_csv("simulation_comparison.csv", index=False)

    print("\n========== COMPARISON ==========")
    print(comparison.to_string(index=False))

    baseline_energy = baseline_summary["total_energy_Wh"]
    adaptive_energy = adaptive_summary["total_energy_Wh"]
    savings = (baseline_energy - adaptive_energy) / baseline_energy * 100

    print(f"\nEnergy savings (adaptive vs fixed): {savings:.2f}%")
    print("\nFiles created:")
    print("- paired_baseline_results.csv")
    print("- adaptive_results.csv")
    print("- adaptation_log.csv")
    print("- simulation_comparison.csv")


if __name__ == "__main__":
    main()
