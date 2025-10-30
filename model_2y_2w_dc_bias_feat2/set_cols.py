import json 

# ## feat2
cols = ['10 m Model Relative U Component',
       '10 m Model Relative V Component', '80 m Temperature',
       '80 m SL Pressure from P & C', '80 m Specific Humidity',
       '1000 mb Model Relative U Component',
       '1000 mb Model Relative V Component',
       '500 mb Model Relative U Component',
       '500 mb Model Relative V Component', 'Surface SL Pressure from P & C',
       '2 m Temperature', '2 m Dew Point Temperature',
       'Surface WRF Down Shortwave Radiation',
       '950 mb Turbulence Kinetic Energy', '1000 mb Turbulence Kinetic Energy',
       '850 mb Turbulence Kinetic Energy', 'Surface CIN', 'Surface CAPE',
       'Surface PBL Height', 'Surface Friction Velocity',
       '500 mb Actual Relative Humidity', '700 mb Actual Relative Humidity',
       '850 mb Actual Relative Humidity', '1000 mb Actual Relative Humidity',
       '500 mb Vert. Comp. Abs Vorticity', '500 mb Geopotential Height',
       '1000 mb Geopotential Height', '850 mb Geopotential Height',
       '500 mb Temperature', '850 mb Temperature', '1000 mb Temperature',
       '700 mb Temperature', 'cos(DOY_Angle)', 'cos(Zenith)', '80 m Model Speed Cubed',
       '975-850 mb Model U Wind Shear', '975-850 mb Model V Wind Shear','bias']

dic = {"cols":cols}

with open("selected_cols/cols_all.json", "w") as f:
    json.dump(dic, f, indent=4)