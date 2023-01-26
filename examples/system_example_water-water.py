import VCSpbx as vcs
from CoolProp.CoolProp import PropsSI as CPPSI
import numpy as np


# Define parameters
cpr_speed = 2000.0
ref = 'R290'
sl = 'INCOMP::MEG[0.5]'
mdot_SL_cond = 0.5
mdot_SL_evap = 0.5
superheat = 6.0
T_SL_hot_in = 30. + 273.15
T_SL_cold_in = -1 + 273.15
h_SL_hot_in = CPPSI('H', 'T', T_SL_hot_in, 'P', 1e5, sl)
h_SL_cold_in = CPPSI('H', 'T', T_SL_cold_in, 'P', 1e5, sl)


# initial guesses
mdot_ref_init = 0.01
pc_init = 18e5
p0_init = 3.0e5
Tc_init = CPPSI('T', 'P', pc_init, 'Q', 0, ref)
h2_init = CPPSI('H', 'P', pc_init, 'T', 60+273.15, ref)
h3_init = CPPSI('H', 'P', pc_init, 'Q', 0, ref)
h4_init = CPPSI('H', 'P', pc_init, 'T', Tc_init-2., ref)
T0_init = CPPSI('T', 'P', p0_init, 'Q', 1, ref)
h5_init = CPPSI('H', 'P', p0_init, 'T', T0_init+superheat, ref)
h1_init = CPPSI('H', 'P', p0_init, 'T', T0_init+superheat+2., ref)


# Define boundaries


# System and components
system = vcs.System('system', tolerance=1e-8)

cpr = vcs.CompressorEfficiency('cpr', system, etaS=0.6, etaV=0.9, stroke=33e-6, speed=cpr_speed, etaEL=0.95)
cond = vcs.CondenserBPHE('cond', system, k=[300.0, 3000.0, 300.0], area=1.0, subcooling=0.1, initial_areafractions= [0.2, 0.6, 0.2])
ihx = vcs.IHX('ihx', system, UA=5.3)
evap = vcs.Evaporator('evap', system, k=[4000., 420.], area=1., superheat=superheat, boundary_switch=True, limit_temp=True)

snkhot = vcs.Sink('snkhot', system)
snkcold = vcs.Sink('snkcold', system)

srchot = vcs.Source('srchot', system, mdot=mdot_SL_cond, p=1.e5, h=h_SL_hot_in)
srccold = vcs.Source('srccold', system, mdot=mdot_SL_evap, p=1.e5, h=h_SL_cold_in)


# Connections
cpr_cond = vcs.Junction('cpr_cond', system, ref, cpr, 'outlet_A', cond, 'inlet_A', mdot_init=mdot_ref_init, p_init=pc_init, h_init=h2_init)
cond_ihx = vcs.Junction('cond_ihx', system, ref, cond, 'outlet_A', ihx, 'inlet_A', mdot_init=mdot_ref_init, p_init=pc_init, h_init=h3_init)
ihx_evap = vcs.Junction('ihx_evap', system, ref, ihx, 'outlet_A', evap, 'inlet_A', mdot_init=mdot_ref_init, p_init=pc_init, h_init=h3_init)
evap_ihx = vcs.Junction('evap_ihx', system, ref, evap, 'outlet_A', ihx, 'inlet_B', mdot_init=mdot_ref_init, p_init=p0_init, h_init=h5_init)
ihx_cpr = vcs.Junction('ihx_cpr', system, ref, ihx, 'outlet_B', cpr, 'inlet_A', mdot_init=mdot_ref_init, p_init=p0_init, h_init=h1_init)

srccold_evap = vcs.Junction('srccold_evap', system, sl, srccold, 'outlet_A', evap, 'inlet_B', mdot_init=mdot_SL_evap, p_init=1e5, h_init=h_SL_cold_in)
evap_snkcold = vcs.Junction('evap_snkcold', system, sl, evap, 'outlet_B', snkcold, 'inlet_A', mdot_init=mdot_SL_evap, p_init=1e5, h_init=h_SL_cold_in)

srchot_cond = vcs.Junction('srchot_cond', system, sl, srchot, 'outlet_A', cond, 'inlet_B', mdot_init=mdot_SL_cond, p_init=1e5, h_init=h_SL_hot_in)
cond_snkhot = vcs.Junction('cond_snkcold', system, sl, cond, 'outlet_B', snkhot, 'inlet_A', mdot_init=mdot_SL_cond, p_init=1e5, h_init=h_SL_hot_in)

system.run(full_output=True)
