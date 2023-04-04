import pandas as pd
import numpy as np
import math

# Meteorological Data from NSRDB
# The required data for solar radiation are imported from SAM
df = pd.read_excel('London2020.xlsx')

T_input = df['Temperature'].values
P_input = df['Pressure'].values / 10  # in kPa
v_input = df['Wind Speed'].values
Ig_input = df['Global Horizontal Irradiance'].values
It_e1 = df['Extra Terrestrial Horizontal Radiation'].values
It_e2 = df['Extra Terrestrial Direct Normal Radiation'].values
It_w1 = df['Global Horizontal Irradiance - Tilted (90°)'].values
It_w2 = df['Direct Normal Irradiance - Tilted (90°)'].values
It_n = df['Diffuse Horizontal Irradiance'].values
It_s = df['Direct Normal Irradiance'].values
hr_no = df['Hour'].values
w_d = df['Wind Direction'].values
sun_rs = df.loc[0:11, 'Sunrise':'Sunset'].values

T_amb = np.zeros((365, 24))
P_amb = np.zeros((365, 24))
v_amb = np.zeros((365, 24))
w_dir = np.zeros((365, 24))
I_G = np.zeros((365, 24))
I_T_e1 = np.zeros((365, 24))
I_T_e2 = np.zeros((365, 24))
I_T_w1 = np.zeros((365, 24))
I_T_w2 = np.zeros((365, 24))
I_T_n = np.zeros((365, 24))
I_T_s = np.zeros((365, 24))
hr_num = np.zeros((365, 24))

for i in range(365):
    T_amb[i, :] = T_input[24*i:24*(i+1)]  # C
    P_amb[i, :] = P_input[24*i:24*(i+1)]  # kPa
    v_amb[i, :] = v_input[24*i:24*(i+1)]  # m/s
    w_dir[i, :] = w_d[24*i:24*(i+1)]  # degree
    I_G[i, :] = Ig_input[24*i:24*(i+1)]  # W/m2
    I_T_e1[i, :] = It_e1[24*i:24*(i+1)]  # W/m2
    I_T_e2[i, :] = It_e2[24*i:24*(i+1)]  # W/m2
    I_T_w1[i, :] = It_w1[24*i:24*(i+1)]  # W/m2
    I_T_w2[i, :] = It_w2[24*i:24*(i+1)]  # W/m2
    I_T_n[i, :] = It_n[24*i:24*(i+1)]  # W/m2
    I_T_s[i, :] = It_s[24*i:24*(i+1)]  # W/m2
    hr_num[i, :] = hr_no[24*i:24*(i+1)]  # hour number

## Greenhouse Construction
# Dimensions

# Constants
W = 3.12  # span width [m]
L = W  # span length [m]
H = 1.9  # gutter height [m]
N_s = 1  # number of spans
beta = 27  # the roof angle [degree]

# Area, Volume, Perimeter
A_gh = N_s * W * L  # m2
H_gtr = 0.5 * W * math.tan(math.radians(beta))  # gutter to ridge height [m]
W_tr = H_gtr / math.sin(math.radians(beta))  # width of tilted roof [m]
V_gh = A_gh * H + N_s * L * 0.5 * W * H_gtr  # m3
Pe_gh = 2 * (W * N_s + L)  # m

# Greenhouse Cover
# "8 mm twinwall polycarbonate"
d_PC = 0.008
k_PC = 0.2
Trans_PC = 0.78
Transl_PC = 0.03
Emis_PC = 0.65
N_PC = 1

# "3 mm glass"
d_GL = 0.003
k_GL = 0.76
Trans_GL = 0.905
Transl_GL = 0.03
Emis_GL = 0.89
N_GL = 1

# "6 mm polyethylene film"
# d_PE=0.006;
# k_PE=0.33;
# Trans_PE=0.75;
# Transl_PE=0.29;
# Emis_PE=0.2;
# N_PE=1;

# Cover
# d_Co=0.008; %m
# k_Co=0.2; %W/m-K
Trans_co = Trans_PC * Trans_GL  # transmissivity
Transl_co = Transl_PC * Transl_GL  # long-wave transmissivity
Emis_co = Emis_PC + Emis_GL * Transl_PC  # emissivity
N_co = 1  # number of layers

# Surfaces
# "East"
A_e1 = L * W_tr  # m2
A_e2 = L * H  # m2

# "West"
A_w1 = L * W_tr  # m2
A_w2 = L * H  # m2

# "North"
A_n1 = W * H_gtr / 2  # m2
A_n2 = W * H  # m2
A_n = A_n1 + A_n2

# "South"
A_s1 = W * H_gtr / 2  # m2
A_s2 = W * H  # m2
A_s = A_s1 + A_s2

# Effective Length
# "Free (Natural) Convection"
L_eff_nc_e1 = W_tr
L_eff_nc_e2 = H
L_eff_nc_w1 = W_tr
L_eff_nc_w2 = H
L_eff_nc_n = H + H_gtr
L_eff_nc_s = H + H_gtr
# "It is provided for forced convection in the following"

## Greenhouse Climate
# Growing Plan "Tomato"
T_gh_d = 22 # C
T_gh_n = 15 # C

# Lighting
n_light = 16
DLI_p = 25 # daily light integral [mol/m2day]
DLI_hour = DLI_p / n_light # average DLI per hour [mol/m2h]
loc_day = np.nonzero(Ig_input)[0] # location of daylight hours in the yearly matrix
sf = np.zeros((365, 24))
for i in range(365):
    n = 1
    for j in range(len(loc_day)):
        if loc_day[j] > (24 * (i - 1)) and loc_day[j] < (24 * i):
            sf[i, n] = loc_day[j] - 24 * (i - 1) # location of daylight hours separately for each day
            n = n + 1

n_day = np.zeros((365, 1))
n_exc_light = np.zeros((365, 1))
loc_start_sl = np.zeros((365, 1))
loc_end_sl = np.zeros((365, 1))

for i in range(365):
    sun_rs[i, 0] = sf[i, 0] # sunrise hour
    sun_rs[i, 1] = np.max(sf[i, :]) # sunset hour
    n_day[i, 0] = np.count_nonzero(sf[i, :]) # number of daylight hours for each day
    n_exc_light[i] = np.abs(n_day[i, 0] - n_light) / 2 # number of excess hours for supplementary lighting
    loc_start_sl[i] = sun_rs[i, 0] - n_exc_light[i] # start point of supplementary lighting
    loc_end_sl[i, 0] = sun_rs[i, 1] + n_exc_light[i] # end point of supplementary lighting

F_lc = 3.5 # light conversion coefficient [micromol/J]
F_hc = 0.6 # heat conversion factor
F_a = 1.1 # lighting allowance factor

# Conduction and Convection
h_a = 3.85 # heat transfer coefficient of insulation air [W/m2K]

# Air Exchange
N_a = 0.7 # 1/hr

# Floor
k_s = 1.33 # soil thermal conductivity [W/mK]
H_s = 3.05 # soil depth [m]

# Evapotranspiration
A_c = 0.7 * A_gh # canopy area [m2]
phi_gh = 0.8 # indoor relative humidity
L_eff_leaf = 0.027 # characteristics length of leaf [m]
LAI = 3 # leaf area index [m2 leaf/m2 canopy]
v_i = 0.2 # inside air speed [m/s]
R_a = 220 * (L_eff_leaf ** 0.2) / (v_i ** 0.8) # aerodynamic resistance of leaf [s/m]
A_p = LAI * A_c # m2
h_v = 2442 # vaporization enthalpy of water [kJ/kg]

# Longwave radiation
sigma = 5.67E-8 # Stephan-Boltzmann constant [W/m2-K4]
Emis_p = 0.9 # plant emissivity
Fp_sky = 1 # view factor between the greenhouse and the sky

## Thermal Calculations
# Initialize arrays with zeros
Q_s = np.zeros((365, 24))
Q_sl = np.zeros((365, 24))
Q_cc = np.zeros((365, 24))
Q_evap = np.zeros((365, 24))
Q_lwr = np.zeros((365, 24))
Q_airex = np.zeros((365, 24))
Q_f = np.zeros((365, 24))
Q_hour = np.zeros((365, 24))
Q_heating_day = np.zeros((365, 1))
Q_cooling_day = np.zeros((365, 1))

T_film_in = np.zeros((365, 24))
T_film_out = np.zeros((365, 24))
k_a_gh = np.zeros((365, 24))
h_i_e1 = np.zeros((365, 24))
h_i_e2 = np.zeros((365, 24))
h_i_w1 = np.zeros((365, 24))
h_i_w2 = np.zeros((365, 24))
h_i_n = np.zeros((365, 24))
h_i_s = np.zeros((365, 24))
w_dir = np.zeros((365, 24))
dt_gh = np.zeros((365, 24))
Cp_a_gh = np.zeros((365, 24))
Gr_e1 = np.zeros((365, 24))
Gr_e2 = np.zeros((365, 24))
Gr_w1 = np.zeros((365, 24))
Gr_w2 = np.zeros((365, 24))
Gr_n = np.zeros((365, 24))
Gr_s = np.zeros((365, 24))
VE = np.zeros((365, 24))
mu_a_gh = np.zeros((365, 24))
rho_a_gh = np.zeros((365, 24))
k_amb = np.zeros((365, 24))
rho_amb = np.zeros((365, 24))
mu_amb = np.zeros((365, 24))
Cp_amb = np.zeros((365, 24))
h_o_e1 = np.zeros((365, 24))
h_o_e2 = np.zeros((365, 24))
h_o_w1 = np.zeros((365, 24))
h_o_w2 = np.zeros((365, 24))
h_o_n = np.zeros((365, 24))
h_o_s = np.zeros((365, 24))
U_e1 = np.zeros((365, 24))
U_e2 = np.zeros((365, 24))
U_w1 = np.zeros((365, 24))
U_w2 = np.zeros((365, 24))
U_n = np.zeros((365, 24))
U_s = np.zeros((365, 24))
Q_cc_e1 = np.zeros((365, 24))
Q_cc_e2 = np.zeros((365, 24))
Q_cc_w1 = np.zeros((365, 24))
Q_cc_w2 = np.zeros((365, 24))
Q_cc_n = np.zeros((365, 24))
Q_cc_s = np.zeros((365, 24))

DLI_sun = np.zeros((365, 24))
DLI_light = np.zeros((365, 24))
W_dot_sl = np.zeros((365, 24))

omega_ps = np.zeros((365, 24))
omega_gh = np.zeros((365, 24))
P_g = np.zeros((365, 24))
R_s = np.zeros((365, 24))
m_dot_moist = np.zeros((365, 24))

Q_lwr_sky = np.zeros((365, 24))
Q_lwr_co = np.zeros((365, 24))
T_co = np.zeros((365, 24))
T_sky = np.zeros((365, 24))


        
        




