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

import numpy as np

for i in range(365):
    for j in range(24):
        
        # Solar Radiation
        # ---------------
        Q_s[i,j] = (I_T_e1[i,j]*N_s*A_e1*Trans_GL + I_T_e2[i,j]*A_e2*Trans_co + I_T_w1[i,j]*N_s*A_w1*Trans_GL +
                    I_T_w2[i,j]*A_w2*Trans_co + I_T_n[i,j]*N_s*A_n*Trans_co + I_T_s[i,j]*N_s*A_s*Trans_co)*3600 # J
        
        # Supplementary Lighting
        # ----------------------
        DLI_sun[i,j] = 2.15E-6*Q_s[i,j]/A_gh # DLI provided by sun [mol/m2day]
        
        if loc_start_sl[i] <= hr_num[i,j] <= loc_end_sl[i]:
            if DLI_sun[i,j] < DLI_hour:
                DLI_light[i,j] = DLI_hour - DLI_sun[i,j] # DLI provided by supplementary lighting [mol/m2day]
                W_dot_sl[i,j] = DLI_light[i,j]*(10**6)*F_a/(F_lc*3600) # power consumed by supplementary lighting [W/m2]
                Q_sl[i,j] = W_dot_sl[i,j]*3600*F_hc*A_gh # J
            
        # Heat Loss by Longwave Radiation
        # -------------------------------
        T_sky[i,j] = 0.055*(T_amb[i,j]+273.15)**1.5-273.15
        if I_G[i,j] > 0:
            T_co[i,j] = T_gh_d/3 + 2*T_amb[i,j]/3
            Q_lwr_sky[i,j] = sigma*Emis_p*Fp_sky*A_gh*Transl_GL*((T_gh_d+273.15)**4-(T_sky[i,j]+273.15)**4) # between plants and sky
            Q_lwr_co[i,j] = sigma*((T_gh_d+273.15)**4-(T_co[i,j]+273.15)**4)*(Emis_GL*N_s*A_e1*((1+np.cos(np.deg2rad(beta)))/2) +
                                Emis_co*A_e2*((1+np.cos(np.deg2rad(90)))/2) + Emis_GL*N_s*A_w1*((1+np.cos(np.deg2rad(beta)))/2) +
                                Emis_co*A_w2*((1+np.cos(np.deg2rad(90)))/2) + Emis_co*N_s*A_n*((1+np.cos(np.deg2rad(90)))/2) +
                                Emis_co*N_s*A_s*((1+np.cos(np.deg2rad(90)))/2))
        else:
            T_co[i,j] = T_gh_n/3 + 2*T_amb[i,j]/3
            Q_lwr_sky[i,j] = 0
            Q_lwr_co[i,j] = sigma*((T_gh_n+273.15)**4 - (T_co[i,j]+273.15)**4)*(Emis_GL*N_s*A_e1*((1+math.cosd(beta))/2) + \
                            Emis_co*A_e2*((1+math.cosd(90))/2) + Emis_GL*N_s*A_w1*((1+math.cosd(beta))/2) + \
                            Emis_co*A_w2*((1+math.cos(90))/2) + Emis_co*N_s*A_n*((1+math.cosd(90))/2) + \
                            Emis_co*N_s*A_s*((1+math.cosd(90))/2))
            
        Q_lwr[i,j] = (Q_lwr_sky[i,j] + Q_lwr_co[i,j])*3600 # J

                            
        ## Conductive and Convective Heat Loss
        # -----------------------------------
        # "Free Convection (Inside)"
        if I_G[i,j]==0:
            T_film_in[i,j]=(T_co[i,j]+T_gh_n)/2
            dt_gh[i,j]=abs(T_gh_n-T_co[i,j])
        else:
            T_film_in[i,j]=(T_co[i,j]+T_gh_d)/2
            dt_gh[i,j]=abs(T_gh_d-T_co[i,j])
        
        VE[i,j]=0.00366041593-0.0000134150047*T_film_in[i,j]+5.08683431E-08*T_film_in[i,j]**2-1.75298514E-10*T_film_in[i,j]**3 #VE_air=0.00366041593-0.0000134150047*T+5.08683431E-08*T^2-1.75298514E-10*T^3 [1/K]
        k_a_gh[i,j]=0.023635147+0.0000756348155*T_film_in[i,j]-2.52191313E-08*T_film_in[i,j]**2 #k_air=0.023635147+0.0000756348155*T-2.52191313E-08*T^2 [W/mK]  
        mu_a_gh[i,j]=0.0000172934955+4.89570039E-08*T_film_in[i,j]-4.28241627E-11*T_film_in[i,j]**2 #mu_air=0.0000172934955+4.89570039E-08*T-4.28241627E-11*T^2 [kg/ms]
        Cp_a_gh[i,j]=(1.05-0.365*((T_film_in[i,j]+273.15)/1000)+0.85*((T_film_in[i,j]+273.15)/1000)**2-0.39*((T_film_in[i,j]+273.15)/1000)**3)*1000 #Cp_air=1000*(1.05-0.365*Theta+0.85*Theta^2-0.39*Theta^3) [J/kgK]--Theta=(T (in Kelvin)/1000)
        rho_a_gh[i,j]=P_amb[i,j]/((8.314/28.97)*(T_film_in[i,j]+273.15)) #density [kg/m3]

        Gr_e1[i,j]=9.81*VE[i,j]*L_eff_nc_e1**3*dt_gh[i,j]/((mu_a_gh[i,j]/rho_a_gh[i,j])**2) #Grashof number
        Gr_e2[i,j]=9.81*VE[i,j]*L_eff_nc_e2**3*dt_gh[i,j]/((mu_a_gh[i,j]/rho_a_gh[i,j])**2)
        Gr_w1[i,j]=9.81*VE[i,j]*L_eff_nc_w1**3*dt_gh[i,j]/((mu_a_gh[i,j]/rho_a_gh[i,j])**2)
        Gr_w2[i,j]=9.81*VE[i,j]*L_eff_nc_w2**3*dt_gh[i,j]/((mu_a_gh[i,j]/rho_a_gh[i,j])**2)
        Gr_n[i,j]=9.81*VE[i,j]*L_eff_nc_n**3*dt_gh[i,j]/((mu_a_gh[i,j]/rho_a_gh[i,j])**2)
        Gr_s[i,j]=9.81*VE[i,j]*L_eff_nc_s**3*dt_gh[i,j]/((mu_a_gh[i,j]/rho_a_gh[i,j])**2)

        h_i_e1[i,j]=(k_a_gh[i,j]/L_eff_nc_e1)*0.1*(Gr_e1[i,j]*mu_a_gh[i,j]*Cp_a_gh[i,j]/k_a_gh[i,j])**0.33 #heat transfer coefficient [W/m2k]
        h_i_e2[i,j]=(k_a_gh[i,j]/L_eff_nc_e2)*0.1*(Gr_e2[i,j]*mu_a_gh[i,j]*Cp_a_gh[i,j]/k_a_gh[i,j])**0.33
        h_i_w1[i,j]=(k_a_gh[i,j]/L_eff_nc_w1)*0.1*(Gr_w1[i,j]*mu_a_gh[i,j]*Cp_a_gh[i,j]/k_a_gh[i,j])**0.33
        h_i_w2[i,j]=(k_a_gh[i,j]/L_eff_nc_w2)*0.1*(Gr_w2[i,j]*mu_a_gh[i,j]*Cp_a_gh[i,j]/k_a_gh[i,j])**0.33
        h_i_n[i,j]=(k_a_gh[i,j]/L_eff_nc_n)*0.1*(Gr_n[i,j]*mu_a_gh[i,j]*Cp_a_gh[i,j]/k_a_gh[i,j])**0.33
        h_i_s[i,j]=(k_a_gh[i,j]/L_eff_nc_s)*0.1*(Gr_s[i,j]*mu_a_gh[i,j]*Cp_a_gh[i,j]/k_a_gh[i,j])**0.33

        # Forced Convection (Outside)
        if w_dir[i,j]<=22.5 and w_dir[i,j]>=337.5 and w_dir[i,j]<=202.5 and w_dir[i,j]>=157.5:
            L_eff_fc_e1=L
            L_eff_fc_e2=L
            L_eff_fc_w1=L
            L_eff_fc_w2=L
            L_eff_fc_n=A_n/(2*H+2*W_tr+W)
            L_eff_fc_s=A_s/(2*H+2*W_tr+W)
        elif w_dir[i,j]<=112.5 and w_dir[i,j]>=67.5 and w_dir[i,j]<=292.5 and w_dir[i,j]>=247.5:
            L_eff_fc_e1=A_e1/(2*(L+W_tr))
            L_eff_fc_e2=A_e2/(2*(L+H))
            L_eff_fc_w1=A_w1/(2*(L+W_tr))
            L_eff_fc_w2=A_w2/(2*(L+H))
            L_eff_fc_n=W
            L_eff_fc_s=W
        else:
            L_eff_fc_e1=A_e1/(2*(L+W_tr))
            L_eff_fc_e2=A_e2/(2*(L+H))
            L_eff_fc_w1=A_w1/(2*(L+W_tr))
            L_eff_fc_w2=A_w2/(2*(L+H))
            L_eff_fc_n=A_n/(2*H+2*W_tr+W)
            L_eff_fc_s=A_s/(2*H+2*W_tr+W)
        
        T_film_out[i,j] = (T_co[i,j] + T_amb[i,j]) / 2
        k_amb[i,j] = 0.023635147 + 0.0000756348155 * T_film_out[i,j] - 2.52191313E-08 * T_film_out[i,j]**2
        mu_amb[i,j] = 0.0000172934955 + 4.89570039E-08 * T_film_out[i,j] - 4.28241627E-11 * T_film_out[i,j]**2
        Cp_amb[i,j] = (1.05 - 0.365 * ((T_film_out[i,j] + 273.15) / 1000) + 0.85 * ((T_film_out[i,j] + 273.15) / 1000)**2 - 0.39 * ((T_film_out[i,j] + 273.15) / 1000)**3) * 1000
        rho_amb[i,j] = P_amb[i,j] / ((8.314 / 28.97) * (T_film_out[i,j] + 273.15))

        h_o_e1[i,j] = (k_amb[i,j] / L_eff_fc_e1) * 0.037 * ((rho_amb[i,j] * v_amb[i,j] * L_eff_fc_e1 / mu_amb[i,j]) ** 0.8) * (mu_amb[i,j] * Cp_amb[i,j] / k_amb[i,j]) ** 0.33 # heat transfer coefficient [W/m2k]
        h_o_e2[i,j] = (k_amb[i,j] / L_eff_fc_e2) * 0.037 * ((rho_amb[i,j] * v_amb[i,j] * L_eff_fc_e2 / mu_amb[i,j]) ** 0.8) * (mu_amb[i,j] * Cp_amb[i,j] / k_amb[i,j]) ** 0.33
        h_o_w1[i,j] = (k_amb[i,j] / L_eff_fc_w1) * 0.037 * ((rho_amb[i,j] * v_amb[i,j] * L_eff_fc_w1 / mu_amb[i,j]) ** 0.8) * (mu_amb[i,j] * Cp_amb[i,j] / k_amb[i,j]) ** 0.33
        h_o_w2[i,j] = (k_amb[i,j] / L_eff_fc_w2) * 0.037 * ((rho_amb[i,j] * v_amb[i,j] * L_eff_fc_w2 / mu_amb[i,j]) ** 0.8) * (mu_amb[i,j] * Cp_amb[i,j] / k_amb[i,j]) ** 0.33
        h_o_n[i,j] = (k_amb[i,j] / L_eff_fc_n) * 0.037 * ((rho_amb[i,j] * v_amb[i,j] * L_eff_fc_n / mu_amb[i,j]) ** 0.8) * (mu_amb[i,j] * Cp_amb[i,j] / k_amb[i,j]) ** 0.33
        h_o_s[i,j] = (k_amb[i,j] / L_eff_fc_s) * 0.037 * ((rho_amb[i,j]))

        # continue at line 379

        U_e1[i,j]=(1/h_i_e1[i,j]+N_GL*d_GL/k_GL+1/h_o_e1[i,j])**(-1) #overall heat transfer coefficient [W/m2k]
        U_e2[i,j]=(1/h_i_e2[i,j]+N_GL*d_GL/k_GL+N_PC*d_PC/k_PC+N_PC/h_a+1/h_o_e2[i,j])**(-1)
        U_w1[i,j]=(1/h_i_w1[i,j]+N_GL*d_GL/k_GL+1/h_o_w1[i,j])**(-1)
        U_w2[i,j]=(1/h_i_w2[i,j]+N_GL*d_GL/k_GL+N_PC*d_PC/k_PC+N_PC/h_a+1/h_o_w2[i,j]+(I_G[i,j]==0)/h_a)**(-1)
        U_n[i,j]=(1/h_i_n[i,j]+N_GL*d_GL/k_GL+N_PC*d_PC/k_PC+N_PC/h_a+1/h_o_n[i,j])**(-1)
        U_s[i,j]=(1/h_i_s[i,j]+N_GL*d_GL/k_GL+N_PC*d_PC/k_PC+N_PC/h_a+1/h_o_s[i,j])**(-1)

        if I_G[i,j]==0:
            Q_cc_e1[i,j]=U_e1[i,j]*N_s*A_e1*(T_gh_n-T_amb[i,j])*3600 #J
            Q_cc_e2[i,j]=U_e2[i,j]*A_e2*(T_gh_n-T_amb[i,j])*3600
            Q_cc_w1[i,j]=U_w1[i,j]*N_s*A_w1*(T_gh_n-T_amb[i,j])*3600
            Q_cc_w2[i,j]=U_w2[i,j]*A_w2*(T_gh_n-T_amb[i,j])*3600
            Q_cc_n[i,j]=U_n[i,j]*N_s*A_n*(T_gh_n-T_amb[i,j])*3600
            Q_cc_s[i,j]=U_s[i,j]*N_s*A_s*(T_gh_n-T_amb[i,j])*3600
        else:
            Q_cc_e1[i,j]=U_e1(i,j)*N_s*A_e1*(T_gh_d-T_amb(i,j))*3600 #J
            Q_cc_e2[i,j]=U_e2(i,j)*A_e2*(T_gh_d-T_amb(i,j))*3600
            Q_cc_w1[i,j]=U_w1(i,j)*N_s*A_w1*(T_gh_d-T_amb(i,j))*3600
            Q_cc_w2[i,j]=U_w2(i,j)*A_w2*(T_gh_d-T_amb(i,j))*3600
            Q_cc_n[i,j]=U_n(i,j)*N_s*A_n*(T_gh_d-T_amb(i,j))*3600
            Q_cc_s[i,j]=U_s(i,j)*N_s*A_s*(T_gh_d-T_amb(i,j))*3600

        Q_cc=Q_cc_e1+Q_cc_e2+Q_cc_w1+Q_cc_w2+Q_cc_n+Q_cc_s

        ## Heat Loss by Air Exchange

        if I_G[i,j]==0:
            Q_airex[i,j]=0.33*N_a*V_gh*(T_gh_n-T_amb[i,j])*3600
        else:
            Q_airex[i,j]=0.33*N_a*V_gh*(T_gh_d-T_amb[i,j])*3600

        ## Heat Loss through the Floor

        if i<=31 and i>=1:
            T_s=8.9 #deep soil temperature [C]
        elif i<=59 and i>=32:
            T_s=8.3
        elif i<=90 and i>=60:
            T_s=7.2
        elif i<=120 and i>=91:
            T_s=6.7
        elif i<=151 and i>=121:
            T_s=6.1
        elif i<=181 and i>=152:
            T_s=6.7
        elif i<=212 and i>=182:
            T_s=7.8
        elif i<=243 and i>=213:
            T_s=9.4
        elif i<=273 and i>=244:
            T_s=10.5
        elif i<=304 and i>=274:
            T_s=11.1
        elif i<=334 and i>=305:
            T_s=11.1
        else:
            T_s=10
        

        if I_G[i,j]==0:
            Q_f[i,j]=(k_s/H_s)*A_gh*(T_gh_n-T_s)*3600 #J  
        else:
            Q_f[i,j]=(k_s/H_s)*A_gh*(T_gh_d-T_s)*3600

        # Heat Loss by Evapotranspiration
        # -------------------------------
        if I_G[i,j]>0:
            P_g[i,j]=0.618543911+0.0443631857*T_gh_d+0.00139030107*T_gh_d**2+0.0000266080784*T_gh_d**3+3.35139063E-07*T_gh_d**4+2.01114464E-09*T_gh_d**5 #P_sat=0.618543911+0.0443631857*T+0.00139030107*T^2+0.0000266080784*T^3+3.35139063E-07*T^4+2.01114464E-09*T^5 [kPa] 
        else:
            P_g[i,j]=0

        omega_ps[i,j]=0.622*P_g[i,j]/(P_amb[i,j]-P_g[i,j])
        omega_gh[i,j]=0.622*phi_gh*P_g[i,j]/(P_amb[i,j]-phi_gh*P_g[i,j])
        R_s[i,j]=200*(1+1/exp(0.05*(I_G[i,j]*Trans_GL-50)))
        rho_a_gh[i,j]=P_amb[i,j]/((8.314/28.97)*(T_amb[i,j]+273.15))
        m_dot_moist[i,j]=A_p*rho_a_gh(i,j)*(omega_ps(i,j)-omega_gh(i,j))/(R_a+R_s(i,j))
        Q_evap[i,j]=m_dot_moist(i,j)*h_v*1000*3600 #J

        ## Hourly Heating Demand
        Q_hour[i,j] = (Q_s[i,j] + Q_sl[i,j] - Q_cc[i,j] - Q_airex[i,j] - Q_f[i,j] - Q_evap[i,j] - Q_lwr[i,j]) / (10**6) #MJ
    
    ## Exporting the Results

    for i in range(365):
        Q_heating_day[i] = sum(Q_hour[i,:] * (Q_hour[i,:] < 0))
        Q_cooling_day[i] = sum(Q_hour[i,:] * (Q_hour[i,:] > 0))

N_miner = 1
Q_miner = 3240 #W
Q_tot_miner = N_miner * Q_miner * 24 * 3600 / (10**6) #MJ
Q_coldest_day = abs(min(Q_heating_day))
Err = Q_tot_miner / 1.05 - Q_coldest_day

Q_heating_year = -sum(Q_heating_day) #MJ
Q_cooling_year = sum(Q_cooling_day) #MJ

