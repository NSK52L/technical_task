
import csv
import numpy as np
from itertools import product

import config
import devices


# ==================================================
# 1. EXPERIMENT CONFIGURATION
# ==================================================

rng = np.random.default_rng(config.random_seed)

execution_locations = ["Edge", "Cloud"]
processing_modes = ["Normal", "Throttled"]
camera_rates = [1, 2, 5, 10]
lamp_states = [True, False]

repetitions = 50
experiment_duration_minutes = 60

output_file = "training_data.csv"


# ==================================================
# 2. GENERATE A BALANCED EXPERIMENT DESIGN
# ==================================================

configurations = list(product(
    execution_locations,
    processing_modes,
    camera_rates,
    lamp_states
))

experiments = configurations * repetitions
rng.shuffle(experiments)

number_of_samples = len(experiments)

data = []


# ==================================================
# 3. GENERATE EXPERIMENTS
# ==================================================

for sample, (location, mode, camera_rate, lamp_on) in enumerate(experiments):

    # Environmental conditions
    storage_temp = rng.uniform(25.0, 48.0)
    ambient_light = rng.uniform(10.0, 500.0)

    # Determine processing capacity
    if location == "Edge":
        if mode == "Normal":
            processing_capacity = config.cv_capacity_edge
        else:
            processing_capacity = config.cv_throttled_capacity_edge

    else:
        if mode == "Normal":
            processing_capacity = config.cv_capacity_cloud
        else:
            processing_capacity = config.cv_throttled_capacity_cloud

    # CPU utilization
    cpu_utilization = min(
        100.0,
        camera_rate / processing_capacity * 100
    )

    # Edge and Cloud CPU usage
    if location == "Edge":
        edge_cpu = cpu_utilization
        cloud_cpu = 0.0
    else:
        edge_cpu = 0.0
        cloud_cpu = cpu_utilization

    # ==================================================
    # 4. POWER CONSUMPTION
    # ==================================================

    edge_p = devices.edge_power(edge_cpu)
    cloud_p = devices.cloud_power(cloud_cpu)

    camera_p = devices.camera_power(True)
    lamp_p = devices.lamp_power(lamp_on)

    # Every captured image is stored
    images_written = camera_rate * experiment_duration_minutes

    duration_seconds = experiment_duration_minutes * 60

    storage_p = devices.storage_power(
        images_written,
        duration_seconds
    )

    # Fan state depends on storage temperature
    fan_on = storage_temp >= config.fan_on_threshold

    fan_p = devices.fan_power(fan_on)
    sensor_p = devices.sensor_power()

    total_p = devices.total_power(
        edge_p,
        cloud_p,
        camera_p,
        lamp_p,
        storage_p,
        fan_p,
        sensor_p
    )

    # Total energy consumed during the experiment
    duration_hours = experiment_duration_minutes / 60

    total_energy_wh = total_p * duration_hours

    # ==================================================
    # 5. PROCESSING PERFORMANCE
    # ==================================================

    processed_rate = min(camera_rate, processing_capacity)

    processed_images = int(
        processed_rate * experiment_duration_minutes
    )

    processing_feasible = (
        processing_capacity >= camera_rate
    )

    # ==================================================
    # 6. DETECTION QUALITY MODEL
    # ==================================================

    # Normalize ambient light between 0 and 1
    light_level = np.clip(ambient_light / 500.0, 0.0, 1.0)

    # Assumed detection probability with lamp OFF
    probability_off = 0.55 + 0.40 * light_level

    # Lamp benefit is greater in darker conditions
    lamp_gain = 0.25 * (1.0 - light_level)

    if lamp_on:
        success_probability = min(
            0.995,
            probability_off + lamp_gain
        )
    else:
        success_probability = probability_off

    # Simulate actual detection outcomes
    successful_detections = rng.binomial(
        processed_images,
        success_probability
    )

    # Observed detection success rate
    if processed_images > 0:
        observed_detection_rate = (
            successful_detections / processed_images
        )
    else:
        observed_detection_rate = 0.0

    expected_detections = (
        processed_rate * success_probability
    )

    observed_detections_per_minute = (
        successful_detections / experiment_duration_minutes
    )

    detection_feasible = (
        expected_detections >= config.min_detections_per_minute
    )

    # ==================================================
    # 7. SYSTEM CONSTRAINTS
    # ==================================================

    power_feasible = total_p <= config.max_power_w

    temperature_feasible = (
        storage_temp <= config.storage_temp_max
    )

    feasible = (
        processing_feasible
        and detection_feasible
        and power_feasible
        and temperature_feasible
    )

    # Energy per processed image
    if processed_images > 0:
        energy_per_image = total_energy_wh / processed_images
    else:
        energy_per_image = 0.0

    # ==================================================
    # 8. STORE EXPERIMENT RESULTS
    # ==================================================

    data.append({

        "sample_id": sample,

        "execution_location": location,
        "processing_mode": mode,
        "camera_rate_images_per_min": camera_rate,
        "lamp_on": lamp_on,

        "storage_temperature_C": round(storage_temp, 2),
        "ambient_light_lux": round(ambient_light, 2),
        "fan_on": fan_on,

        "processing_capacity_images_per_min": processing_capacity,
        "cpu_utilization_percent": round(cpu_utilization, 2),

        "edge_power_W": round(edge_p, 3),
        "cloud_power_W": round(cloud_p, 3),
        "camera_power_W": round(camera_p, 3),
        "lamp_power_W": round(lamp_p, 3),
        "storage_power_W": round(storage_p, 3),
        "fan_power_W": round(fan_p, 3),
        "sensor_power_W": round(sensor_p, 3),

        "total_power_W": round(total_p, 3),
        "total_energy_Wh": round(total_energy_wh, 3),

        "processed_rate_images_per_min": round(processed_rate, 3),
        "processed_images": processed_images,

        "detection_success_probability_assumed": round(
            success_probability, 4
        ),

        "successful_detections": successful_detections,

        "observed_detection_success_rate": round(
            observed_detection_rate, 4
        ),

        "expected_detections_per_min": round(
            expected_detections, 4
        ),

        "observed_detections_per_min": round(
            observed_detections_per_minute, 4
        ),

        "processing_feasible": processing_feasible,
        "detection_feasible": detection_feasible,
        "power_feasible": power_feasible,
        "temperature_feasible": temperature_feasible,
        "feasible": feasible,

        "energy_per_image_Wh": round(energy_per_image, 6)
    })


# ==================================================
# 9. SAVE DATASET
# ==================================================

fieldnames = list(data[0].keys())

with open(output_file, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(data)


# ==================================================
# 10. DATASET SUMMARY
# ==================================================

feasible_count = sum(row["feasible"] for row in data)

infeasible_count = number_of_samples - feasible_count

print("\n========== DATASET GENERATION ==========")

print("Total experiments:", number_of_samples)
print("Unique configurations:", len(configurations))
print("Repetitions per configuration:", repetitions)

print("Feasible experiments:", feasible_count)
print("Infeasible experiments:", infeasible_count)

print(
    "Feasible percentage:",
    round(feasible_count / number_of_samples * 100, 2),
    "%"
)

print(
    "Average detection success rate:",
    round(
        np.mean([
            row["observed_detection_success_rate"]
            for row in data
        ]),
        3
    )
)

print("\nDataset saved to:", output_file)
