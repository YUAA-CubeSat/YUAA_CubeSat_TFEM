import numpy as np
import pandas as pd
import pint; from pint import Quantity; u=pint.UnitRegistry()

"""Units convention:
pint Quantities are used only in initialization and public interfaces, 
all runtime internals are pure floats in SI.
Exceptions:
- SGP4 positions, which are in units of km
"""

# ~~~~~~~~~ PHYSICAL CONSTANTS ~~~~~~~~~~
SB_SIGMA = 5.670e-8 * u.W/(u.m**2 * u.degK**4) # Stefan-Boltzman law

# ~~~~~~~~~ MATERIAL PROPERTIES ~~~~~~~~~
CU_EMISSIVITY = 0.03 # https://www.engineeringtoolbox.com/emissivity-coefficients-d_447.html, using "electroplated copper"
CU_ABSORPTANCE = 0.18 # https://www.engineeringtoolbox.com/solar-radiation-absorbed-materials-d_1568.html, using "copper, polished"
SOLAR_CELL_EMISSIVITY = 0.85 # Taken from datasheet
SOLAR_CELL_ABSORPTANCE = 0.88 # Taken from datasheet
SOLAR_PANEL_SC_AREA_FRACTION = 6.91*3.97*4 / (10*20) # Using cell dims from datasheet and 4 cells per long side
SOLAR_PANEL_EMISSIVITY = SOLAR_PANEL_SC_AREA_FRACTION*SOLAR_CELL_EMISSIVITY+(1-SOLAR_PANEL_SC_AREA_FRACTION)*CU_EMISSIVITY
SOLAR_PANEL_ABSORPTANCE = SOLAR_PANEL_SC_AREA_FRACTION*SOLAR_CELL_ABSORPTANCE+(1-SOLAR_PANEL_SC_AREA_FRACTION)*CU_ABSORPTANCE
FR4_EMISSIVITY = 0.95 # http://www.solarmirror.com/fom/fom-serve/cache/43.html Eyeballing from values for black plastics
FR4_ABSORPTANCE = 0.95 # http://www.solarmirror.com/fom/fom-serve/cache/43.html
AL_6061_SPEC_HEAT_CAP = 897*u.J/(u.kg*u.degK) # Wikipedia
AL_EMISSIVITY = 0.03 # http://www.solarmirror.com/fom/fom-serve/cache/43.html
AL_ABSORPTANCE = 0.09 # http://www.solarmirror.com/fom/fom-serve/cache/43.html
AL_HEAT_CONDUCTIVITY = 400 * u.W/(u.m**2 * u.degK) # [1] TODO - refine
PB_SPEC_HEAT_CAP = 127 * u.J/(u.kg*u.degK)

# ~~~~~~~~ CUBESAT PROPERTIES ~~~~~~~~~~~
CUBESAT_TOTAL_EXT_AREA = 10*u.cm*10*u.cm*(1+1+2*4)
CUBESAT_MASS = 3.6*u.kg
CUBESAT_TIP_MASS = 0.8*u.kg
CUBESAT_TIP_MASS_EXT_AREA = 6.441*u.cm * np.pi * 3.9*u.cm + (np.pi * (6.441/2*u.cm)**2)
CUBESAT_ROTI_XY = 0.0147*u.kg*u.m**2
CUBESAT_ROTI_Z = 0.00415*u.kg*u.m**2
CUBESAT_POWER_CONSUMPTION = (274+206*.6+50+74+50+877)*u.mW # Assuming system power level 4 w/o active detumbling, excluding EPS battery heating, and assuming 40% of the TCV power draw gets radiated as RF

# ~~~~~~~~ ASTRONOMICAL PROPERTIES ~~~~~~
EARTH_RADIUS = 6371*u.km
EARTH_YEAR = pd.Timedelta(days=365.256363004)
EARTH_SEMIMAJOR_AXIS = 149598023*u.km
EARTH_AXIAL_TILT_RAD = np.radians(23.4392811)
ECLIPTIC_TO_EQUATORIAL_R = \
    np.array([
        [1, 0, 0],
        [0, np.cos(EARTH_AXIAL_TILT_RAD), -np.sin(EARTH_AXIAL_TILT_RAD)],
        [0, np.sin(EARTH_AXIAL_TILT_RAD), np.cos(EARTH_AXIAL_TILT_RAD)]
    ])
EARTH_ALBEDO_FACTOR = 0.19 # https://smallsat.uga.edu/images/documents/papers/Preliminary_Thermal_Analysis_of_Small_Satellites.pdf
EARTH_IR_NORMAL_POWER = 218*u.W/(u.m**2) # https://smallsat.uga.edu/images/documents/papers/Preliminary_Thermal_Analysis_of_Small_Satellites.pdf
LEO_TYP_SOLAR_NORMAL_POWER = 1368*u.W/(u.m**2) # https://smallsat.uga.edu/images/documents/papers/Preliminary_Thermal_Analysis_of_Small_Satellites.pdf
ISS_TLE = (
    '1 25544U 98067A   25249.87397102  .00012937  00000-0  23296-3 0  9996',
    '2 25544  51.6325 262.1963 0004213 309.4705  50.5911 15.50156361527830'
)