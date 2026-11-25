
import csv
import numpy as np
from collections import deque

import config
import environment
import devices



# Initialization

rng = np.random.default_rng(config.random_seed)

dt = config.time_step_seconds              
total_steps = int(config.sim_hours * 3600 / dt)

camera_rate = config.camera_rate_images_per_minute
edge_capacity = config.cv_capacity_edge

# Fixed baseline configuration
camera_on = True
lamp_on = True
cloud_cpu = 0.0

# Initial system state
storage_temperature = config.storage_temp_initial
fan_on = False

# Image management
capture_credit = 0.0
processing_credit = 0.0
queue_size = 0

total_captured = 0
total_processed = 0
total_successful = 0

# Energy and power monitoring
total_energy_wh = 0.0
max_power_observed = 0.0
power_budget_violations = 0

# Rolling 60-second detection window
window_steps = int(60 / dt)
detection_window = deque(maxlen=window_steps)
requirement_violations = 0

output_file = "baseline_results.csv"

# Basic configuration checks
if dt <= 0:
    raise ValueError("Time step must be positive.")

if 60 % dt != 0:
    raise ValueError("Time step must divide 60 seconds exactly.")

if edge_capacity <= 0:
    raise ValueError("Edge processing capacity must be positive.")

print("Starting baseline simulation...")
print("Number of simulation steps:", total_steps)

# CSV File

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "time_seconds",
        "solar_power_W",
        "ambient_temperature_C",
        "captured_images",
        "processed_images",
        "successful_detections",
        "queue_size",
        "storage_temperature_C",
        "fan_on",
        "edge_power_W",
        "cloud_power_W",
        "camera_power_W",
        "lamp_power_W",
        "storage_power_W",
        "fan_power_W",
        "sensor_power_W",
        "total_power_W",
        "grid_power_W",
        "energy_Wh",
        "rolling_detections"
    ])


    # Main loop

    for step in range(total_steps):

        time_seconds = step * dt
        hour = (time_seconds / 3600) % 24


        solar = environment.solar_power(hour)
        ambient_temp = environment.ambient_temperature(hour, rng)


        images_written = 0
        if camera_on:
            capture_credit += camera_rate * dt / 60
            images_written = int(capture_credit)
            capture_credit -= images_written
        total_captured += images_written   
        queue_size += images_written # Every captured image is stored immediately
        processing_credit += edge_capacity * dt / 60
        images_to_process = min(queue_size, int(processing_credit))
        queue_size -= images_to_process
        
        
        processing_credit -= images_to_process
        total_processed += images_to_process
        if lamp_on:
            success_probability = config.lamp_on_success_prob
        else:
            success_probability = config.lamp_off_success_prob
            
        successful_detections = 0
        for _ in range(images_to_process):
            if rng.random() < success_probability:
                successful_detections += 1
        total_successful += successful_detections


        detection_window.append(successful_detections)
        rolling_detections = sum(detection_window)
        if len(detection_window) == window_steps:
            if rolling_detections < config.min_detections_per_minute:
                requirement_violations += 1


        edge_cpu = min(100.0, camera_rate / edge_capacity * 100)
        edge_p = devices.edge_power(edge_cpu)
        cloud_p = devices.cloud_power(cloud_cpu)
        camera_p = devices.camera_power(camera_on)
        lamp_p = devices.lamp_power(lamp_on)
        storage_p = devices.storage_power(images_written, dt)
        fan_on = devices.update_fan_state(storage_temperature, fan_on)
        fan_p = devices.fan_power(fan_on)
        sensor_p = devices.sensor_power()
        total_p = devices.total_power(edge_p, cloud_p, camera_p, lamp_p, storage_p, fan_p, sensor_p)
        max_power_observed = max(max_power_observed, total_p)

        if total_p > config.max_power_w:
            power_budget_violations += 1


        grid_p = max(0.0, total_p - solar)


        energy_this_step = total_p * dt / 3600
        total_energy_wh += energy_this_step


        storage_temperature = devices.update_storage_temp(
            current_temp=storage_temperature,
            ambient_temp=ambient_temp,
            storage_power_val=storage_p,
            fan_on=fan_on,
            time_step_seconds=dt
        )
        

        writer.writerow([
            time_seconds,
            solar,
            ambient_temp,
            images_written,
            images_to_process,
            successful_detections,
            queue_size,
            storage_temperature,
            fan_on,
            edge_p,
            cloud_p,
            camera_p,
            lamp_p,
            storage_p,
            fan_p,
            sensor_p,
            total_p,
            grid_p,
            total_energy_wh,
            rolling_detections
        ])



# Final Results

print("\n========== BASELINE RESULTS ==========")

print("Simulated duration:", config.sim_hours, "hours")

print("Total captured images:", total_captured)
print("Total processed images:", total_processed)
print("Successful detections:", total_successful)
print("Images still waiting:", queue_size)

print(
    "Final storage temperature:",
    round(storage_temperature, 2), "°C"
)

print(
    "Maximum system power:",
    round(max_power_observed, 2), "W"
)

print(
    "Power budget violations:",
    power_budget_violations
)

print(
    "Total energy consumption:",
    round(total_energy_wh, 2), "Wh"
)

print(
    "Total energy consumption:",
    round(total_energy_wh / 1000, 4), "kWh"
)

print(
    "Detection requirement violations:",
    requirement_violations
)

print("\nResults saved to:", output_file)
