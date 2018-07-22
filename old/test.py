# Polynomial Regression

# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import time 
import json
import urllib.request

with urllib.request.urlopen("https://alte-rs.ddnss.de/weather/processed/history.json") as url:
    data1 = json.loads(url.read().decode())

df = pd.DataFrame(data=data1)
df['time'] = pd.to_datetime(df['time'])
df['temp'] = pd.to_numeric(df['temp'])
df['pressure'] = pd.to_numeric(df['pressure'])
df['humidity'] = pd.to_numeric(df['humidity'])

df2 = df.set_index('time')



series = df2.rolling('30Min',min_periods=20).mean()
total =[df2,series]
conc= pd.concat(total, axis=1)
print(conc)