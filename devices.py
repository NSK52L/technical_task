import config 

# Raspberry Pi power model

def edge_power(cpu_utilization): 
	cpu_utilization = max(0, min(100, cpu_utilization)) # limit cpu to 100
	
	idle_power = config.rpi_idle_power
	max_power = config.rpi_max_power
	
	return idle_power + (max_power - idle_power) * cpu_utilization / 100	

# Cloud VM Power Model

def cloud_power(cpu_utilization):
	cpu_utilization = max(0, min(100, cpu_utilization)) # limit cpu to 100
	
	idle_power = config.cloud_idle_power
	max_power = config.cloud_max_power
	
	return idle_power + (max_power - idle_power) * cpu_utilization / 100

# Camera power

def camera_power(camera_on = True):
	if camera_on :
		return config.camera_power
	return 0.0

# Lamp power
def lamp_power(lamp_on):
	if lamp_on:
		return config.lamp_power
	return 0.0

# Fan
def fan_power(fan_on):
	if fan_on:
		return config.fan_power
	return 0.0	

# Sensor
def sensor_power():
	return config.sensor_power

# Network storage power
def storage_power(images_written, time_step_seconds):
	idle_power = config.storage_idle_power
	
	energy_per_image = config.storage_write_energy_wh_per_image
	write_energy = images_written * energy_per_image
	time_step_hours = time_step_seconds / 3600
	if time_step_hours <= 0:
		raise ValueError("Time step must be positive.")

	return idle_power + (write_energy / time_step_hours)  

# Total 
def total_power(edge_p, cloud_p, camera_p, lamp_p, storage_p, fan_p, sensor_p):

    return (edge_p + cloud_p + camera_p + lamp_p + storage_p + fan_p + sensor_p)

# Storage temp
def update_storage_temp(current_temp, ambient_temp, storage_power_val, fan_on, time_step_seconds):
	if time_step_seconds <= 0:
		raise ValueError("Time step must be positive.")
	
	heat_power = config.heat_fraction * storage_power_val
	natural_cooling = ( (current_temp - ambient_temp) / config.thermal_resistance )  # Natural heat transfer to the environment in W
	fan_cooling = (config.fan_cooling_power if fan_on else 0.0 ) # Additional fan cooling in W
	net_heat = heat_power - natural_cooling - fan_cooling # Net heat entering the storage thermal system W
	
	temp_change = (net_heat * time_step_seconds / config.thermal_capacitance)
	
	return current_temp + temp_change

# Fan controller
def update_fan_state(current_temp, fan_state):
	if current_temp >= config.fan_on_threshold:
		return True
	elif current_temp <= config.fan_off_threshold:
		return False
	return fan_state


