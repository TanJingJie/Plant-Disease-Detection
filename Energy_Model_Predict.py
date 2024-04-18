# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:56:24 2024

@author: Sailing587
"""
import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import linear_model

# Define connection parameters
server = 'DESKTOP-I1OSNLL\MSSQLSERVER01'
database = 'PlantDetection'



temp1 = 30
temp2 = 23

# Load data
data = pd.read_csv(r"D:\FYP Uni\GreenHouse Data\Energy Model.csv")

# Prepare data
X_for_area = data[['HVAC Temp', 'Fan Speed']]
X_for_area = X_for_area.values
Y_Area1 = data['Area 1']
Y_Area2 = data['Area 2']
Y_Area3 = data['Area 3']

#Data for best efficiency
Area_1 = data['Area 1']
X_for_efficiency = data[['Area 1', 'Area 2', 'Area 3']]
X_for_efficiency = X_for_efficiency.values
Y_HVAC = data['HVAC Temp']
Y_Fan = data['Fan Speed']

# Y_HVAC_Energy = data['HVAC energy']
# Y_Fan_Energy = data['Fan energy']


# Train the model
regr = linear_model.LinearRegression(fit_intercept=False)


# Function to predict Areas temperature
def get_area1_temperature(desired_HVAC_temperature, desired_fanspeed):
    # Split data into training and testing sets
    X__for_area_train, X__for_area_test, Y_Area1_train, Y_Area1_test = train_test_split(X_for_area, Y_Area1, test_size=0.2, random_state=42)
    
    regr.fit(X_for_area, Y_Area1) 
    # Make prediction using the trained model
    predicted_area1_temperature = regr.predict([[desired_HVAC_temperature, desired_fanspeed]])
    return predicted_area1_temperature[0]

def get_area2_temperature(desired_HVAC_temperature, desired_fanspeed):
    # Split data into training and testing sets
    X__for_area_train, X__for_area_test, Y_Area2_train, Y_Area2_test = train_test_split(X_for_area, Y_Area2, test_size=0.2, random_state=42)
    
    regr.fit(X_for_area, Y_Area2) 
    # Make prediction using the trained model
    predicted_area2_temperature = regr.predict([[desired_HVAC_temperature, desired_fanspeed]])
    return predicted_area2_temperature[0]

def get_area3_temperature(desired_HVAC_temperature, desired_fanspeed):
    # Split data into training and testing sets
    X_for_area_train, X_for_area_test, Y_Area3_train, Y_Area3_test = train_test_split(X_for_area, Y_Area3, test_size=0.2, random_state=42)
    regr.fit(X_for_area, Y_Area3) 
    # Make prediction using the trained model
    predicted_area3_temperature = regr.predict([[desired_HVAC_temperature, desired_fanspeed]])
    return predicted_area3_temperature[0]

# Function for best combination
def get_area_temperature_HVAC(desired_area_temperature):
    # Split data into training and testing sets
    X_for_efficiency_train, X_for_efficiency_test, Y_HVAC_train, Y_HVAC_test = train_test_split(X_for_efficiency, Y_HVAC, test_size=0.2, random_state=42)
    regr.fit(X_for_efficiency, Y_HVAC) 
    # Make prediction using the trained model
    predicted_HVAC_temperature = regr.predict([[desired_area_temperature, desired_area_temperature, desired_area_temperature]])
    return predicted_HVAC_temperature[0]

def get_area_temperature_Fan(desired_area_temperature):
    # Split data into training and testing sets
    X_for_efficiency_train, X_for_efficiency_test, Y_Fan_train, Y_Fan_test = train_test_split(X_for_efficiency, Y_Fan, test_size=0.2, random_state=42)
    regr.fit(X_for_efficiency, Y_Fan) 
    # Make prediction using the trained model
    predicted_Fan_speed = regr.predict([[desired_area_temperature, desired_area_temperature, desired_area_temperature]])
    return predicted_Fan_speed[0]

# User input for desired temperature and fan speed and mass of crops
mass = float(input("Enter mass of crops in chamber in kg: "))
desired_HVAC_temperature = float(input("Enter desired HVAC temperature: "))
desired_fanspeed = float(input("Enter fan speed: "))
desired_area_temperature =  float(input("Enter desired area temperature: "))


def energy (desired_HVAC_temperature, desired_fanspeed):
    
    Q1 = 0.28*121.5*(temp1 - temp2)/1000
    Q2 = (mass * 3.94) / 3600
    Q3 = 39.4 #Lighting 
    Q4 = 3 * 91.125 * 2 * (temp1 - temp2 ) / 3600  

    HVAC_Q = ((Q1 + Q2 + Q3 + Q4)/4)

    Fan_Q = (100 * 2 * 1 * desired_fanspeed)  /(1000) 

    Total_Q = HVAC_Q + Fan_Q

    return Total_Q
    
    

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Area_1, Y_HVAC, Y_Fan)

ax.set_xlabel('Area Temperature')
ax.set_ylabel('HVAC Temp')
ax.set_zlabel('Fan Speed')

plt.title('HVAC Temp and Fan Speed vs Area Temperature')

plt.show()

# Predict HVAC temperature
predicted_Area1Temp = get_area1_temperature(desired_HVAC_temperature, desired_fanspeed)
predicted_Area2Temp = get_area2_temperature(desired_HVAC_temperature, desired_fanspeed)
predicted_Area3Temp = get_area3_temperature(desired_HVAC_temperature, desired_fanspeed)
predicted_Energy = energy(desired_HVAC_temperature, desired_fanspeed)
predicted_HVAC_Temperature=get_area_temperature_HVAC(desired_area_temperature)
predicted_Fan_Speed=get_area_temperature_Fan(desired_area_temperature)
predicted_Best_Efficiency = energy(predicted_HVAC_Temperature, predicted_Fan_Speed)
energy_cost_hour = (predicted_Best_Efficiency * 32.47 )/100
energy_cost_yearly = energy_cost_hour * 24 * 365
print(f"Predicted Area 1 temperature for desired temperature {desired_HVAC_temperature}°C and fan speed {desired_fanspeed}: {predicted_Area1Temp}")
print(f"Predicted Area 2 temperature for desired temperature {desired_HVAC_temperature}°C and fan speed {desired_fanspeed}: {predicted_Area2Temp}")
print(f"Predicted Area 3 temperature for desired temperature {desired_HVAC_temperature}°C and fan speed {desired_fanspeed}: {predicted_Area2Temp}")
print(f"Predicted Energy consumption for desired temperature {desired_HVAC_temperature}°C and fan speed {desired_fanspeed}: {predicted_Energy} kW/hr")
print(f"Predicted best HVAC temperature and fan speed for area temperature {desired_area_temperature}°C : {predicted_HVAC_Temperature} °C and {predicted_Fan_Speed}m/s")
print(f"Predicted Energy consumption for desired temperature {predicted_HVAC_Temperature}°C and fan speed {predicted_Fan_Speed}: {predicted_Best_Efficiency} kW/hr")
print(f"Predicted cost per hour using predicted best efficiency: ${energy_cost_hour}")
print(f"Predicted cost per year using predicted best efficiency: ${energy_cost_yearly}")