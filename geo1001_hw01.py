#-- GEO1001.2020--hw01
#-- Louise Spekking
#-- 4256778

import pandas as pd
from statistics import variance
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# read data
sensor_a = pd.read_excel('./hw_1_data/HEAT - A_final.xls', skiprows = 3).drop([0])
sensor_b = pd.read_excel('./hw_1_data/HEAT - B_final.xls', skiprows = 3).drop([0])
sensor_c = pd.read_excel('./hw_1_data/HEAT - C_final.xls', skiprows = 3).drop([0])
sensor_d = pd.read_excel('./hw_1_data/HEAT - D_final.xls', skiprows = 3).drop([0])
sensor_e = pd.read_excel('./hw_1_data/HEAT - E_final.xls', skiprows = 3).drop([0])

#correct data types 
to_numeric_columns = ['Direction ‚ True', 'Wind Speed',
       'Crosswind Speed', 'Headwind Speed', 'Temperature', 'Globe Temperature',
       'Wind Chill', 'Relative Humidity', 'Heat Stress Index', 'Dew Point',
       'Psychro Wet Bulb Temperature', 'Station Pressure',
       'Barometric Pressure', 'Altitude', 'Density Altitude',
       'NA Wet Bulb Temperature', 'WBGT', 'TWL', 'Direction ‚ Mag']

sensor_a[to_numeric_columns] = sensor_a[to_numeric_columns].apply(pd.to_numeric)
sensor_a['FORMATTED DATE-TIME'] = pd.to_datetime(sensor_a['FORMATTED DATE-TIME'])

sensor_b[to_numeric_columns] = sensor_b[to_numeric_columns].apply(pd.to_numeric)
sensor_b['FORMATTED DATE-TIME'] = pd.to_datetime(sensor_b['FORMATTED DATE-TIME'])

sensor_c[to_numeric_columns] = sensor_c[to_numeric_columns].apply(pd.to_numeric)
sensor_c['FORMATTED DATE-TIME'] = pd.to_datetime(sensor_c['FORMATTED DATE-TIME'])

sensor_d[to_numeric_columns] = sensor_d[to_numeric_columns].apply(pd.to_numeric)
sensor_d['FORMATTED DATE-TIME'] = pd.to_datetime(sensor_d['FORMATTED DATE-TIME'])

sensor_e[to_numeric_columns] = sensor_e[to_numeric_columns].apply(pd.to_numeric)
sensor_e['FORMATTED DATE-TIME'] = pd.to_datetime(sensor_e['FORMATTED DATE-TIME'])

# check for NaN values
if sensor_a.isnull().values.ravel().sum() and sensor_b.isnull().values.ravel().sum() and sensor_c.isnull().values.ravel().sum() and sensor_d.isnull().values.ravel().sum() and sensor_e.isnull().values.ravel().sum():
    print('No NaN values in dataset')
else:
    print('NaN values in dataset')

#calculate mean,m variance and std
sensor_list = [sensor_a, sensor_b, sensor_c, sensor_d, sensor_e]
name_list = ['sensor_a','sensor_b', 'sensor_c', 'sensor_d', 'sensor_e']

i = 0
mean_dict = {} 
for sensor in sensor_list:
    mean_list = [] 
    sensor_name = name_list[i]
    for measurement in to_numeric_columns:
        mean = sensor[measurement].mean()
        mean_list.append(mean)
    mean_dict.update({sensor_name : mean_list})
    i += 1

mean_df = pd.DataFrame.from_dict(mean_dict, orient = 'index', columns = to_numeric_columns)
mean_df2 = mean_df.transpose()
mean_df2.to_csv('mean_variables.csv')
print('Mean values per sensor per measurent', mean_df2)

i = 0
var_dict = {} 
for sensor in sensor_list:
    var_list = [] 
    sensor_name = name_list[i]
    for measurement in to_numeric_columns:
        var = variance(sensor[measurement])
        var_list.append(var)
    var_dict.update({sensor_name : var_list})
    i += 1

var_df = pd.DataFrame.from_dict(var_dict, orient = 'index', columns = to_numeric_columns)
var_df2 = var_df.transpose()
print('Variance per sensor per measurent', var_df2)
var_df2.to_csv('variance_variables.csv')

i = 0
std_dict = {} 
for sensor in sensor_list:
    std_list = [] 
    sensor_name = name_list[i]
    for measurement in to_numeric_columns:
        stdev = np.std(sensor[measurement])
        std_list.append(stdev)
    std_dict.update({sensor_name : std_list})
    i += 1

std_df = pd.DataFrame.from_dict(std_dict, orient = 'index', columns = to_numeric_columns)
std_df2 = std_df.transpose()
print('Standard deviation per sensor per measurent', std_df2)
std_df2.to_csv('std_variables.csv')

#histograms for temperature 
Temperature = pd.concat([sensor_a['Temperature'], sensor_b['Temperature'], sensor_c['Temperature'],
                       sensor_d['Temperature'], sensor_e['Temperature']], 
                       axis=1, keys=['A', 'B', 'C', 'D', 'E']).dropna()

ax = sns.histplot(data=Temperature, bins = 5, palette="Set3", multiple="stack")
ax.set(xlabel = 'Temperature')
plt.savefig('Histogram of temperature per sensor with 5 bins')
plt.show()

ax = sns.histplot(data=Temperature, bins = 50, palette="Set3", multiple="stack")
ax.set(xlabel = 'Temperature')
plt.savefig('Histogram of temperature per sensor with 50 bins')
plt.show()

#frequency polygons
ax = sns.displot(data=Temperature, element = 'poly', palette="Set3",
                 multiple="stack", bins = 27)
ax.set(xlabel = 'Temperature')
plt.savefig('polygon hist temperature')
plt.show()

#boxplots 
wind_direction = pd.concat([sensor_a['Direction ‚ True'], sensor_b['Direction ‚ True'], 
                        sensor_c['Direction ‚ True'], sensor_d['Direction ‚ True'], 
                        sensor_e['Direction ‚ True']], 
                       axis=1, keys=['A', 'B', 'C', 'D', 'E'])

Wind_speed = pd.concat([sensor_a['Wind Speed'], sensor_b['Wind Speed'], sensor_c['Wind Speed'],
                       sensor_d['Wind Speed'], sensor_e['Wind Speed']], 
                       axis=1, keys=['A', 'B', 'C', 'D', 'E'])

fig , axes = plt.subplots(nrows = 1, ncols = 3, figsize=(20,5))
plt.sca(axes[0])
ax = sns.boxplot(data=Wind_speed, palette="Set3")
ax.set(xlabel = 'Sensor', ylabel = 'Wind Speed',
      title = 'Boxplot of the wind speed in m/s per sensor')

plt.sca(axes[1])
ax = sns.boxplot(data=wind_direction, palette="Set3")
ax.set(xlabel = 'Sensor', ylabel = 'Wind Direction',
      title = 'Boxplot of the wind direction in degrees per sensor') 

plt.sca(axes[2])
ax = sns.boxplot(data=Temperature, palette="Set3")
ax.set(xlabel = 'Sensor', ylabel = 'Temperature',
       title = 'Boxplot of temperature in degrees C per sensor')
 
plt.savefig('Boxplots') 
plt.show()

# A2

# pmf
pmf_a = sensor_a['Temperature'].value_counts().sort_index() / len(sensor_a['Temperature'])
pmf_b = sensor_b['Temperature'].value_counts().sort_index() / len(sensor_b['Temperature'])
pmf_c = sensor_c['Temperature'].value_counts().sort_index() / len(sensor_c['Temperature'])
pmf_d = sensor_d['Temperature'].value_counts().sort_index() / len(sensor_d['Temperature'])
pmf_e = sensor_e['Temperature'].value_counts().sort_index() / len(sensor_e['Temperature'])

fig , axes = plt.subplots(nrows = 1, ncols = 5, figsize=(20,5))
plt.sca(axes[0])
plt.bar(pmf_a.index, pmf_a.values)
plt.title('Sensor A')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.xticks(np.arange(min(sensor_a.Temperature), max(sensor_a.Temperature)+1, 5.0))

plt.sca(axes[1])
plt.bar(pmf_b.index, pmf_b.values)
plt.title('Sensor B')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.xticks(np.arange(min(sensor_b.Temperature), max(sensor_b.Temperature)+1, 5.0))

plt.sca(axes[2])
plt.bar(pmf_c.index, pmf_c.values)
plt.title('Sensor C')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.xticks(np.arange(min(sensor_c.Temperature), max(sensor_c.Temperature)+1, 5.0))
plt.yticks([0.000, 0.005, 0.010, 0.015, 0.020])

plt.sca(axes[3])
plt.bar(pmf_d.index, pmf_d.values)
plt.title('Sensor D')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.xticks(np.arange(min(sensor_d.Temperature), max(sensor_d.Temperature)+1, 5.0))

plt.sca(axes[4])
plt.bar(pmf_e.index, pmf_e.values)
plt.title('Sensor E')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.xticks(np.arange(min(sensor_e.Temperature), max(sensor_e.Temperature)+1, 5.0))

fig.suptitle('PMF of the temperature per sensor', fontsize = 15, y = 1.08)
plt.tight_layout()
plt.savefig('PMF of temperature per sensor') 
plt.show()

#PDF

fig , axes = plt.subplots(nrows = 1, ncols = 5, figsize=(20,5))
plt.sca(axes[0])
ax = sensor_a['Temperature'].plot(kind='density')
ax.set(xlabel = 'Temperature')
plt.title('Sensor A')
plt.ylabel('Density')

plt.sca(axes[1])
ax = sensor_b['Temperature'].plot(kind='density')
ax.set(xlabel = 'Temperature')
plt.title('Sensor B')
plt.ylabel('Density')

plt.sca(axes[2])
ax = sensor_c['Temperature'].plot(kind='density')
ax.set(xlabel = 'Temperature')
plt.title('Sensor C')
plt.ylabel('Density')

plt.sca(axes[3])
ax = sensor_d['Temperature'].plot(kind='density')
ax.set(xlabel = 'Temperature')
plt.title('Sensor D')
plt.ylabel('Density')

plt.sca(axes[4])
ax = sensor_e['Temperature'].plot(kind='density')
ax.set(xlabel = 'Temperature')
plt.title('Sensor E')
plt.ylabel('Density')

plt.tight_layout()
fig.suptitle('PDF of the temperature per sensor', fontsize = 15, y = 1.08)

plt.savefig('PDF of temperature per sensor') 
plt.show()

#CDF
fig , axes = plt.subplots(nrows = 1, ncols = 5, figsize=(20,5))
plt.sca(axes[0])
a1 = plt.hist(x=sensor_a['Temperature'], bins=50, cumulative=True, density = True, alpha=0.7, rwidth=0.85)
plt.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
plt.title('Sensor A')
plt.xlabel('Temperature')
plt.ylabel('CDF')

plt.sca(axes[1])
b1 = plt.hist(x=sensor_b['Temperature'], bins=50, cumulative=True, density = True, alpha=0.7, rwidth=0.85)
plt.plot(b1[1][1:]-(b1[1][1:]-b1[1][:-1])/2,b1[0], color='k')
plt.title('Sensor B')
plt.xlabel('Temperature')
plt.ylabel('CDF')

plt.sca(axes[2])
c1 = plt.hist(x=sensor_c['Temperature'], bins=50, cumulative=True, density = True, alpha=0.7, rwidth=0.85)
plt.plot(c1[1][1:]-(c1[1][1:]-c1[1][:-1])/2,c1[0], color='k')
plt.title('Sensor C')
plt.xlabel('Temperature')
plt.ylabel('CDF')

plt.sca(axes[3])
d1 = plt.hist(x=sensor_d['Temperature'], bins=50, cumulative=True, density = True ,alpha=0.7, rwidth=0.85)
plt.plot(d1[1][1:]-(d1[1][1:]-a1[1][:-1])/2,d1[0], color='k')
plt.title('Sensor D')
plt.xlabel('Temperature')
plt.ylabel('CDF')

plt.sca(axes[4])
e1 = plt.hist(x=sensor_d['Temperature'], bins=50, cumulative=True, density = True, alpha=0.7, rwidth=0.85)
plt.plot(e1[1][1:]-(e1[1][1:]-e1[1][:-1])/2,e1[0], color='k')
plt.title('Sensor E')
plt.xlabel('Temperature')
plt.ylabel('CDF')

plt.tight_layout()
fig.suptitle('CDF of the temperature per sensor', fontsize = 15, y = 1.08)
plt.savefig('CDF of temperature per sensor') 
plt.show()

# windspeed PDF and KDE
fig , axes = plt.subplots(nrows = 1, ncols = 5, figsize=(20,5))

plt.sca(axes[0])
ax = sns.distplot(Wind_speed.A, kde = True, norm_hist = True)
ax.set(xlabel = 'Wind Speed')
plt.title('Sensor A')

plt.sca(axes[1])
ax = sns.distplot(Wind_speed.B, kde = True, norm_hist = True)
ax.set(xlabel = 'Wind Speed')
plt.title('Sensor B')

plt.sca(axes[2])
ax = sns.distplot(Wind_speed.C, kde = True, norm_hist = True)
ax.set(xlabel = 'Wind Speed')
plt.title('Sensor C')

plt.sca(axes[3])
ax = sns.distplot(Wind_speed.D, kde = True, norm_hist = True)
ax.set(xlabel = 'Wind Speed')
plt.title('Sensor D')

plt.sca(axes[4])
ax = sns.distplot(Wind_speed.E, kde = True, norm_hist = True)
ax.set(xlabel = 'Wind Speed')
plt.title('Sensor E')

plt.tight_layout()
fig.suptitle('KDE and PDF of wind speed per sensor', fontsize = 15, y = 1.08)
plt.savefig('KDE and PDF of wind speed per sensor') 
plt.show()

#A3

Wet_Bulb_Globe = pd.concat([sensor_a['WBGT'], sensor_b['WBGT'], 
                        sensor_c['WBGT'], sensor_d['WBGT'], 
                        sensor_e['WBGT']], 
                       axis=1, keys=['A', 'B', 'C', 'D', 'E']).dropna()
Crosswind_speed = pd.concat([sensor_a['Crosswind Speed'], sensor_b['Crosswind Speed'], 
                        sensor_c['Crosswind Speed'], sensor_d['Crosswind Speed'], 
                        sensor_e['Crosswind Speed']], 
                       axis=1, keys=['A', 'B', 'C', 'D', 'E']).dropna()

#make correlations 
sensor_list = ['A', 'B', 'C', 'D', 'E']
pearson_dict_W = {} 
spearman_dict_W = {} 

while len(sensor_list) > 1:
    sensor1 = sensor_list[0]
    for sensor2 in sensor_list:
        if sensor1 != sensor2: 
            pearson, _ = stats.pearsonr(Wet_Bulb_Globe[sensor1], Wet_Bulb_Globe[sensor2])
            key = str(sensor1) + 'x' + str(sensor2)
            pearson_dict_W[key] = pearson

            spearman, _ = stats.spearmanr(Wet_Bulb_Globe[sensor1], Wet_Bulb_Globe[sensor2])
            key = str(sensor1) + 'x' + str(sensor2)
            spearman_dict_W[key] = spearman
    sensor_list.remove(sensor1)
        
sensor_list = ['A', 'B', 'C', 'D', 'E']
pearson_dict_T = {} 
spearman_dict_T = {} 

while len(sensor_list) > 1:
    sensor1 = sensor_list[0]
    for sensor2 in sensor_list:
        if sensor1 != sensor2: 
            pearson, _ = stats.pearsonr(Temperature[sensor1], Temperature[sensor2])
            key = str(sensor1) + 'x' + str(sensor2)
            pearson_dict_T[key] = pearson
            
            spearman, _ = stats.spearmanr(Temperature[sensor1], Temperature[sensor2])
            key = str(sensor1) + 'x' + str(sensor2)
            spearman_dict_T[key] = spearman
    sensor_list.remove(sensor1)

sensor_list = ['A', 'B', 'C', 'D', 'E']
pearson_dict_C = {} 
spearman_dict_C = {} 

while len(sensor_list) > 1:
    sensor1 = sensor_list[0]
    for sensor2 in sensor_list:
        if sensor1 != sensor2: 
            pearson, _ = stats.pearsonr(Crosswind_speed[sensor1], Crosswind_speed[sensor2])
            key = str(sensor1) + 'x' + str(sensor2)
            pearson_dict_C[key] = pearson

            spearman, _ = stats.spearmanr(Crosswind_speed[sensor1], Crosswind_speed[sensor2])
            key = str(sensor1) + 'x' + str(sensor2)
            spearman_dict_C[key] = spearman
    sensor_list.remove(sensor1)

sensor_list = ['A', 'B', 'C', 'D', 'E']

x = list(spearman_dict_C.keys())
y_cs = list(spearman_dict_C.values())
y_ts = list(spearman_dict_T.values())
y_ws = list(spearman_dict_W.values())
y_cp = list(pearson_dict_C.values())
y_tp = list(pearson_dict_T.values())
y_wp = list(pearson_dict_W.values())

fig , axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10,5))

plt.sca(axes[0])
ax = sns.scatterplot(x, y_cs)
ax = sns.scatterplot(x, y_ts)
ax = sns.scatterplot(x, y_ws)
ax.set(ylabel = 'Correlation coeffiecient')
plt.title('Spearman correlation')

plt.sca(axes[1])
ax = sns.scatterplot(x, y_cp)
ax = sns.scatterplot(x, y_tp)
ax = sns.scatterplot(x, y_wp)
ax.set(ylabel = 'Correlation coeffiecient')
plt.title('Pearson correlation')

plt.legend(labels=['Crosswind Speed', 'Temperature', 'Wet Bulb Globe Temperature'], frameon=False)

plt.tight_layout()
fig.suptitle('Correlations between sensors', fontsize = 15, y = 1.08)

plt.savefig('Pearson and Spearman correlations') 
plt.show()

# A4

#CDF 

fig , axes = plt.subplots(nrows = 1, ncols = 5, figsize=(20,5))
plt.sca(axes[0])
a1 = plt.hist(x=sensor_a['Wind Speed'], bins=50, cumulative=True, density = True, alpha=0.7, rwidth=0.85)
plt.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
plt.title('Sensor A')
plt.xlabel('Wind Speed')
plt.ylabel('CDF')

plt.sca(axes[1])
b1 = plt.hist(x=sensor_b['Wind Speed'], bins=50, cumulative=True, density = True, alpha=0.7, rwidth=0.85)
plt.plot(b1[1][1:]-(b1[1][1:]-b1[1][:-1])/2,b1[0], color='k')
plt.title('Sensor B')
plt.xlabel('Wind Speed')
plt.ylabel('CDF')

plt.sca(axes[2])
c1 = plt.hist(x=sensor_c['Wind Speed'], bins=50, cumulative=True, density = True, alpha=0.7, rwidth=0.85)
plt.plot(c1[1][1:]-(c1[1][1:]-c1[1][:-1])/2,c1[0], color='k')
plt.title('Sensor C')
plt.xlabel('Wind Speed')
plt.ylabel('CDF')

plt.sca(axes[3])
d1 = plt.hist(x=sensor_d['Wind Speed'], bins=50, cumulative=True, density = True ,alpha=0.7, rwidth=0.85)
plt.plot(d1[1][1:]-(d1[1][1:]-d1[1][:-1])/2,d1[0], color='k')
plt.title('Sensor D')
plt.xlabel('Wind Speed')
plt.ylabel('CDF')

plt.sca(axes[4])
e1 = plt.hist(x=sensor_d['Wind Speed'], bins=50, cumulative=True, density = True, alpha=0.7, rwidth=0.85)
plt.plot(e1[1][1:]-(e1[1][1:]-e1[1][:-1])/2,e1[0], color='k')
plt.title('Sensor E')
plt.xlabel('Wind Speed')
plt.ylabel('CDF')

plt.tight_layout()
fig.suptitle('CDF of the wind speed per sensor', fontsize = 15, y = 1.08)
plt.savefig('CDF of Wind Speed per sensor')
plt.show()

#intervals 
sensor_names = [sensor_a, sensor_b, sensor_c, sensor_d, sensor_e]
lower_list = []
upper_list = []

for sensor in sensor_names: 
    mean = np.mean(sensor['Temperature'])
    std = np.std(sensor['Temperature'])
    sample_size = len(sensor['Temperature'])
    lower, upper = stats.t.interval(0.95, sample_size-1, loc=mean, scale= std)

    lower_list.append(lower)
    upper_list.append(upper)

df_temp_intervals = pd.DataFrame([sensor_list, lower_list, upper_list]).transpose()
df_temp_intervals.columns = ['Sensor', 'Lower boundary', 'Upper boundary']
df_temp_intervals.to_csv('confidence_intervals_temperatures.csv')
print('The confidance intervals for temperature per sensor with a confidance level of 95% are', df_temp_intervals)


lower_list_WS = []
upper_list_WS = []

for sensor in sensor_names: 
    mean = np.mean(sensor['Wind Speed'])
    std = np.std(sensor['Wind Speed'])
    sample_size = len(sensor['Wind Speed'])
    lower, upper = stats.t.interval(0.95, sample_size-1, loc=mean, scale= std)
    if lower < 0: 
        lower = 0
        
    lower_list_WS.append(lower)
    upper_list_WS.append(upper)
    
df_windspeed_intervals = pd.DataFrame([sensor_list, 
                                  lower_list_WS, 
                                  upper_list_WS]).transpose()
df_windspeed_intervals.columns = ['Sensor', 'Lower boundary', 
                                  'Upper boundary']

df_windspeed_intervals.to_csv('confidence_intervals_windspeed_95.csv')
print('The confidance intervals for wind speed per sensor with a confidance level of 95% are', df_windspeed_intervals)

lower_list_WS = []
upper_list_WS = []

for sensor in sensor_names: 
    mean = np.mean(sensor['Wind Speed'])
    std = np.std(sensor['Wind Speed'])
    sample_size = len(sensor['Wind Speed'])
    lower, upper = stats.t.interval(0.70, sample_size-1, loc=mean, scale= std)
    if lower < 0: 
        lower = 0
        
    lower_list_WS.append(lower)
    upper_list_WS.append(upper)
    
df_windspeed_intervals_70 = pd.DataFrame([sensor_list, 
                                  lower_list_WS, 
                                  upper_list_WS]).transpose()
df_windspeed_intervals_70.columns = ['Sensor', 'Lower boundary', 
                                  'Upper boundary']

df_windspeed_intervals_70.to_csv('confidence_intervals_windspeed_70.csv')
print('The confidance intervals for wind speed per sensor with a confidance level of 70% are', df_windspeed_intervals_70)

#Hypothesis test

inverted_sensor_list = ['E', 'D', 'C', 'B', 'A']

hypothesis_dict_T = {}
for sensor1 in inverted_sensor_list: 
    if sensor1 != 'A':
        values_list = []
        index = inverted_sensor_list.index(sensor1)
        sensor2 = inverted_sensor_list[index + 1]
        t, p = stats.ttest_rel(Temperature[sensor1], Temperature[sensor2])
        values_list.append(t)
        values_list.append(p)
        key = sensor1 + '-' + sensor2
        hypothesis_dict_T[key] = values_list
hypo_test = pd.DataFrame.from_dict(hypothesis_dict_T).transpose()
hypo_test.columns = ['t-statistic', 'p-value']
print('hypothesis test for temperature', hypo_test)

#hypothesis test wind speed
Wind_speed2 = Wind_speed.dropna()

hypothesis_dict_W = {}
for sensor1 in inverted_sensor_list:
    if sensor1 != 'A':
        print(sensor1)
        values_list = []
        index = inverted_sensor_list.index(sensor1)
        sensor2 = inverted_sensor_list[index + 1]
        t, p = stats.ttest_rel(Wind_speed2[sensor1], Wind_speed2[sensor2])
        values_list.append(t)
        values_list.append(p)
        key = sensor1 + '-' + sensor2
        hypothesis_dict_W[key] = values_list
        
hypo_test_W = pd.DataFrame.from_dict(hypothesis_dict_W).transpose()
hypo_test_W.columns = ['t-statistic', 'p-value']
hypo_test_W.to_csv('hypothesis_test_W.csv')
print('Hypothesis test for wind speed', hypo_test_W)

# Bonus 

#goup by date and take the mean of the temperature data
sensor_a_t = sensor_a[['FORMATTED DATE-TIME', 'Temperature']]
grouped_a = sensor_a_t.set_index('FORMATTED DATE-TIME').groupby(pd.Grouper(freq='D')).mean()

sensor_b_t = sensor_b[['FORMATTED DATE-TIME', 'Temperature']]
grouped_b = sensor_b_t.set_index('FORMATTED DATE-TIME').groupby(pd.Grouper(freq='D')).mean()

sensor_c_t = sensor_c[['FORMATTED DATE-TIME', 'Temperature']]
grouped_c = sensor_c_t.set_index('FORMATTED DATE-TIME').groupby(pd.Grouper(freq='D')).mean()

sensor_d_t = sensor_d[['FORMATTED DATE-TIME', 'Temperature']]
grouped_d = sensor_d_t.set_index('FORMATTED DATE-TIME').groupby(pd.Grouper(freq='D')).mean()

sensor_e_t = sensor_e[['FORMATTED DATE-TIME', 'Temperature']]
grouped_e = sensor_e_t.set_index('FORMATTED DATE-TIME').groupby(pd.Grouper(freq='D')).mean()

#lowest temp days
print('Lowest temperature days sensor A', grouped_a.nsmallest(2, 'Temperature'))
print('Lowest temperature days sensor B', grouped_b.nsmallest(2, 'Temperature'))
print('Lowest temperature days sensor C', grouped_c.nsmallest(2, 'Temperature'))
print('Lowest temperature days sensor D', grouped_d.nsmallest(2, 'Temperature'))
print('Lowest temperature days sensor E', grouped_e.nsmallest(2, 'Temperature'))

#highest temperature days 
print('Highest temperature days sensor A', grouped_a.nlargest(2, 'Temperature'))
print('Highest temperature days sensor B', grouped_b.nlargest(2, 'Temperature'))
print('Highest temperature days sensor C', grouped_c.nlargest(2, 'Temperature'))
print('Highest temperature days sensor D', grouped_d.nlargest(2, 'Temperature'))
print('Highest temperature days sensor E', grouped_e.nlargest(2, 'Temperature'))
