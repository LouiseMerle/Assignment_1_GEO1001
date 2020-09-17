## Variables Description

### FORMATTED DATE-TIME
Date and time in format [YYYY-MM-DD HH:MM:SS]
Sampling frequency: 20 min

### Wind direction [deg]
Compass heading in true or magnetic

### Wind speed [m/s]
Wind Speed is the measurement of the wind passing through the impeller. For greatest accuracy, point the back of the Kestrel directly into the wind.

### Crosswind Speed [m/s]
Crosswind uses the internal compass and a user selected heading to calculate the crosswind component of the full wind.

### Headwind Speed [m/s]
Headwind uses the internal compass and a user selected heading or target direction to calculate the headwind component of the full wind.

### Temperature [deg C]
Ambient Temperature is the temperature measured at the thermistor. For best results, ensure the thermistor is not exposed to direct sunlight and is exposed to good airflow.

### Globe Temperature [deg C]
Globe Temperature is defined as the temperature measured inside a 6-inch copper globe painted black. On the Kestrel HST, the temperature inside the 1-inch|25 mm globe is converted to the equivalent temperature for a standard globe. The closest equivalence will be obtained with airflow greater than 2.2 mph|1 m/s.


### Wind chill [deg C]
Wind Chill is a calculated value of the perceived temperature based on temperature and wind speed

### Relative humidity [%]
Relative Humidity is the amount of moisture currently held by the air as a percentage of the total possible moisture that the air could hold.

### Heat Stress Index [deg C]
Heat Index is a calculated value of the perceived temperature based on temperature and relative humidity.

### Dew Point [deg C]
Dew Point is the temperature at which water vapor will begin to condense out of the air.

### Psychro Wet Bulb Temperature [deg C]
Wet Bulb is the lowest temperature that can be reached in the existing environment by cooling through evaporation. Wet Bulb is always equal to or lower than ambient temperature.

### Station pressure [mb]
Station Pressure (Absolute Pressure) is the pressure exerted by the earth’s atmosphere at any given point.

### Barometric pressure [mb]
Barometric Pressure is the local station (or absolute) pressure with the pressure differential associated with the locations altitude above sea level subtracted. An accurate reading depends on an accurate initial altitude input and unchanging altitude while measuring.

### Altitude [m]
Altitude is the change in vertical distance associated with a change in atmospheric pressure.
An accurate reading depends on an accurate initial barometric pressure input and stable barometric pressure while measuring.

### Density Altitude [m]
Density Altitude is the altitude at which the density of the theoretical standard atmospheric conditions (ISA) would match the actual local air density.

### NA Wet Bulb Tempterature [deg C]
Natural Wet Bulb Temperature is a measure of evaporative cooling in an environment with unforced, naturally occurring air flow.

### WBGT [deg C]
Wet Bulb Globe Temperature is a measure of human heat stress resulting from the combination of effects due
to temperature, humidity, wind speed (wind chill), and visible and radiant heat. Outdoor WBGT is calculated from a weighted sum of Natural Wet Bulb Temperature, Globe Temperature and dry bulb Temperature.

### TWL [w/m^2]
Thermal Work Limit is a measure of the heat energy a person can dissipate from their surface area in Watts per square meter (w/m2). For more information on TWL see http://www.tandfonline.com/doi/ abs/10.1080/104732202753438261.

### Direction, Mag [deg]

[Postprocess rutines](https://github.com/gsclara/KestrelHeatStress)
