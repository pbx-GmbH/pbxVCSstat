import VCSpbx as vcs
from CoolProp.CoolProp import PropsSI as CPPSI
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def generate_parameter_list():
    cpr_range = np.array([1000., 1250., 1500., 1750., 2000., 2500., 3000., 4000., 5000., 6000., 7000., 8000.])
    # cpr_range = np.array([1000., 1250., 1500., 1750., 2000.])

    T_SL_hot_in_range = np.arange(20, 50.1, 1.) + 273.15
    # T_SL_hot_in_range = np.arange(20, 22.1, 1) + 273.15
    h_SL_hot_in_range = np.array([CPPSI('H', 'T', t, 'P', 1e5, 'INCOMP::MEG[0.5]') for t in T_SL_hot_in_range])

    T_SL_cold_in_range = np.arange(-10, 10, 1) + 273.15
    # T_SL_cold_in_range = np.arange(-10, -6, 1) + 273.15
    h_SL_cold_in_range = np.array([CPPSI('H', 'T', t, 'P', 1e5, 'INCOMP::MEG[0.5]') for t in T_SL_cold_in_range])

    return_list = list()
    skip_flag = False
    for h_SL_cold in h_SL_cold_in_range:
        for h_SL_hot in h_SL_hot_in_range:
            for cpr_speed in cpr_range:
                return_list.append([h_SL_cold, h_SL_hot, cpr_speed])
            cpr_range = cpr_range[::-1]
        h_SL_hot_in_range = h_SL_hot_in_range[::-1]

    return return_list


# parameter setting

cpr_speed = 2000.0  # rpm
cpr_stroke = 33.0e-6
ref = 'R290'
SL = 'INCOMP::MEG[0.5]'
superheat = 4.0  # K
subcooling = 0.5  # K
p_SL = 1e5  # Pa

k_cond = [300., 800., 300.]
area_cond = 0.565

k_evap = [500., 100.]
area_evap = 1.

# boundary conditions
T_SL_cold_in = -10.0 + 273.15  # K
T_SL_hot_in = 30.0 + 273.15  # K
h_SL_cold_in = CPPSI('H', 'T', T_SL_cold_in, 'P', p_SL, SL)
h_SL_hot_in = CPPSI('H', 'T', T_SL_hot_in, 'P', p_SL, SL)
mdot_SL_cold = 0.35  # kg/s
mdot_SL_hot = 0.35  # kg/s

# initial guesses
mdot_ref_init = 0.01
pc_init = 12e5
Tc_init = CPPSI('T', 'P', pc_init, 'Q', 0, ref)
h2_init = CPPSI('H', 'P', pc_init, 'T', 70+273.15, ref)
h3_init = CPPSI('H', 'P', pc_init, 'Q', 0, ref)
h4_init = CPPSI('H', 'P', pc_init, 'T', Tc_init-2., ref)
p0_init = 2.9e5
T0_init = CPPSI('T', 'P', p0_init, 'Q', 1, ref)
h5_init = CPPSI('H', 'P', p0_init, 'T', T0_init+superheat, ref)
h1_init = CPPSI('H', 'P', p0_init, 'T', T0_init+superheat+2., ref)

initial_areafraction_cond = [0.09, 0.9, 0.01]
initial_areafraction_evap = [0.2, 0.8]

# Instantiate components
system = vcs.System(id='system', tolerance=1., fun_tol=1.)
cpr = vcs.CompressorEfficiency(id='cpr', system=system, etaS=0.645, etaV=0.82, etaEL=0.775, stroke=cpr_stroke, speed=cpr_speed)
cond = vcs.CondenserCounterflow(id='cond', system=system, k=k_cond, area=area_cond, subcooling=subcooling, initial_areafractions=initial_areafraction_cond)
ihx = vcs.IHX(id='ihx', system=system, UA=2.3)
evap = vcs.EvaporatorCounterflow(id='evap', system=system, k=k_evap, area=area_evap, superheat=superheat, initial_areafractions=initial_areafraction_evap)

snkcold = vcs.Sink(id='snkcold', system=system)
srccold = vcs.Source(id='srccold', system=system, mdot=mdot_SL_cold, p=p_SL, h=h_SL_cold_in)

snkhot = vcs.Sink(id='snkhot', system=system)
srchot = vcs.Source(id='srchot', system=system, mdot=mdot_SL_hot, p=p_SL, h=h_SL_hot_in)

# connections
cpr_cond = vcs.Junction(id='cpr_cond', system=system, medium=ref, upstream_component=cpr, upstream_port_id='outlet_A', downstream_component=cond, downstream_port_id='inlet_A', mdot_init=mdot_ref_init, p_init=pc_init, h_init=h2_init)
cond_ihx = vcs.Junction(id='cond_ihx', system=system, medium=ref, upstream_component=cond, upstream_port_id='outlet_A', downstream_component=ihx, downstream_port_id='inlet_A', mdot_init= mdot_ref_init, p_init=pc_init, h_init=h3_init)
ihx_evap = vcs.Junction(id='ihx_evap', system=system, medium=ref, upstream_component=ihx, upstream_port_id='outlet_A', downstream_component=evap, downstream_port_id='inlet_A', mdot_init= mdot_ref_init, p_init=p0_init, h_init=h4_init)
evap_ihx = vcs.Junction(id='evap_ihx', system=system, medium=ref, upstream_component=evap, upstream_port_id='outlet_A', downstream_component=ihx, downstream_port_id='inlet_B', mdot_init=mdot_ref_init, p_init=p0_init, h_init=h5_init)
ihx_cpr = vcs.Junction(id='ihx_cpr', system=system, medium=ref, upstream_component=ihx, upstream_port_id='outlet_B', downstream_component=cpr, downstream_port_id='inlet_A', mdot_init=mdot_ref_init, p_init=p0_init, h_init=h1_init)

srccold_evap = vcs.Junction(id='srccold_evap', system=system, medium=SL, upstream_component=srccold, upstream_port_id='outlet_A', downstream_component=evap, downstream_port_id='inlet_B', mdot_init=mdot_SL_cold, p_init=p_SL, h_init=h_SL_cold_in)
evap_snkcold = vcs.Junction(id='evap_snkcold', system=system, medium=SL, upstream_component=evap, upstream_port_id='outlet_B', downstream_component=snkcold, downstream_port_id='inlet_A', mdot_init=mdot_SL_cold, p_init=p_SL, h_init=h_SL_cold_in)

srchot_cond = vcs.Junction(id='srchot_cond', system=system, medium=SL, upstream_component=srchot, upstream_port_id='outlet_A', downstream_component=cond, downstream_port_id='inlet_B', mdot_init=mdot_SL_hot, p_init=p_SL, h_init=h_SL_hot_in)
cond_snkhot = vcs.Junction(id='cond_snkhot', system=system, medium=SL, upstream_component=cond, upstream_port_id='outlet_B', downstream_component=snkhot, downstream_port_id='inlet_A', mdot_init=mdot_SL_hot, p_init=p_SL, h_init=h_SL_hot_in)

system.initialize()
system.run(full_output=True)

## T_SL_in_hot approximation
# T_hot_in_target = -5 + 273.15 -.001
# T_range = np.arange(T_SL_hot_in, T_hot_in_target, -1)
# for t in T_range:
#     system.run()
#     h = CPPSI('H', 'T', t, 'P', 1e5, SL)
#     srchot.set_enthalpy(h)
#     print(t)
#
# system.run(full_output=True)
#
# T_SL_in_cold approximation
# T_cold_in_target = -35 + 273.15 -.001
# T_range = np.arange(T_SL_hot_in, T_cold_in_target, -1)
# for t in T_range:
#     system.run()
#     h = CPPSI('H', 'T', t, 'P', 1e5, SL)
#     srccold.set_enthalpy(h)
#     print(t)
#
# system.run(full_output=True)


# cpr.set_speed(1100)
# system.run(full_output=True)
#
# parameter_values = generate_parameter_list()
# parameter_handles = [srccold.set_enthalpy, srchot.set_enthalpy, cpr.set_speed]
# system.parameter_variation(parameters=parameter_values, parameter_handles=parameter_handles)

# plot the root
p_range = np.arange(8e5, 42.1e5, 1e5)
x = np.zeros(7)
x[0] = cond.p
x[1] = cond.T_SL1
x[2] = cond.T_SL2
x[3] = cond.T_SLo
x[4] = cond.f_dsh
x[5] = cond.f_cond
x[6] = cond.f_sc

val_list = list()
for p in p_range:
    vec = np.zeros(7)
    vec[0] = p
    vec[1:] = x[1:].copy()
    val_list.append(vec)

x_dat = list()
y_dat = list()
function_list = ['dshEB1', 'dshEB2', 'condhEB1', 'condEB2','scEB1', 'scEB2', 'fA']
for vals in val_list:
    x_dat.append(vals[0])
    y_dat.append(cond.model(vals))

x_dat, y_dat = np.array(x_dat), np.array(y_dat)
for i in range(len(function_list)):
    ax = plt.subplot(2, 4, i+1)
    ax.plot(x_dat, y_dat[:, i])
    ax.plot(x_dat, np.zeros_like(x_dat), 'r-')
    ax.title.set_text(function_list[i])

plt.show()
