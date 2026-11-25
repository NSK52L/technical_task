import math
import numpy as np

# Solar Power Production

def solar_power(hour):
	max_solar_power = 120.0

	# No sunlight between 6PM and 6AM
	if hour < 6 or hour > 18:	
		 return 0.0

	# Normalize time between sunrize and sunset
	normalized_time = (hour - 6) / 12

	# Sinusoidal model of solar production
	power = max_solar_power * math.sin(math.pi * normalized_time)

	return max(0.0, power)

# Ambient temperature

def ambient_temperature(hour, rng):
	avg_temperature = 22.0
	temp_amplitude = 5.0

	daily_variation = temp_amplitude * math.sin(2 * math.pi * (hour - 9) / 24)
	noise = rng.normal(0, 0.3) # random variation

	return  avg_temperature + daily_variation + noise

# Ambient illumination

def ambient_light(hour, rng):
	if hour < 6 or hour > 18:
		base_light = 10.0 # night light in lux
	else :
		normalized_time = (hour - 6) / 12
		base_light = 500.0 * math.sin(math.pi * normalized_time) # daylight in lux
	noise = rng.normal(0,5) # random variation

	return max(0.0, base_light + noise)
