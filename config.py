# Simulation configuration
sim_hours = 24
time_step_seconds = 10
random_seed = 42

# System requirements 
max_power_w = 1000.0
min_detections_per_minute = 1

# Camera config
camera_rate_images_per_minute = 5
camera_power = 2 

# Raspberry Pi 
rpi_idle_power = 3.0
rpi_max_power = 8.0

# Cloud (VM)
cloud_idle_power = 30.0
cloud_max_power = 60.0

# Computer Vision processing capacity
cv_capacity_edge = 8.0
cv_capacity_cloud = 30.0

cv_throttled_capacity_edge = 4.0
cv_throttled_capacity_cloud = 15.0

# Devices 
lamp_power = 10.0
fan_power = 3.0
storage_idle_power = 5.0
sensor_power = 1.0

# Temp Config
storage_temp_initial = 30.0
fan_on_threshold = 45.0
fan_off_threshold = 40.0
storage_temp_max = 50.0

# Thermal model
heat_fraction = 0.8                 # Fraction of storage power converted to heat
thermal_capacitance = 500.0         # J/°C
thermal_resistance = 4.0            # °C/W
fan_cooling_power = 3.0             # W

# Detection
lamp_on_success_prob = 0.98
lamp_off_success_prob = 0.75

# Storage 
max_queue_size = 1000
storage_write_energy_wh_per_image = 0.0005 
