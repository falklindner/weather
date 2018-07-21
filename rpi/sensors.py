from sense_hat import SenseHat
import json

sense = SenseHat()
sense.clear()

pressure = sense.get_pressure()
temperature = sense.get_temperature()
humidity = sense.get_humidity()

print temperature,pressure,humidity
