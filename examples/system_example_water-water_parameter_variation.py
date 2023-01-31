import VCSpbx as vcs
from CoolProp.CoolProp import PropsSI as CPPSI
import numpy as np
import pandas as pd


def generate_parameter_list():
    cpr_range = np.array([1000., 1250., 1500., 1750., 2000., 2500., 3000., 4000., 5000., 6000., 7000., 8000.])

    T_SL_hot_in_range = np.arange(20, 50.1, 1) + 273.15
    h_SL_hot_in_range = np.array([CPPSI('H', 'T', t, 'P', 1e5, 'INCOMP::MEG[0.5]') for t in T_SL_hot_in_range])

    T_SL_cold_in_range = np.arange(-10, 10, 1) + 273.15
    h_SL_cold_in_range = np.array([CPPSI('H', 'T', t, 'P', 1e5, 'INCOMP::MEG[0.5]') for t in T_SL_cold_in_range])

    return_list = list()
    return_list.append([
        {'component': cpr, 'parameter': 'speed', 'value': cpr_range[0]},
        {'component': srchot, 'parameter': 'h', 'value': h_SL_hot_in_range[0]},
        {'component': srccold, 'parameter': 'h', 'value': h_SL_cold_in_range[0]}
    ])
    skip_flag = False
    for h_SL_cold in h_SL_hot_in_range:
        return_list.append([{'component': srccold, 'parameter': 'h', 'value': h_SL_cold}])
        for h_SL_hot in h_SL_cold_in_range:
            return_list.append([{'component': srchot, 'parameter': 'h', 'value': h_SL_hot}])
            for cpr_speed in cpr_range:
                if skip_flag:
                    skip_flag = False
                    continue
                return_list.append([{'component': cpr, 'parameter': 'speed', 'value': cpr_speed}])
            cpr_range = cpr_range[::-1]
            skip_flag = True
        h_SL_hot_in_range = h_SL_hot_in_range[::-1]

    return return_list


# parameter setting

cpr_speed = 4200.0  # rpm
cpr_stroke = 15.0e-6
ref = 'R290'
SL = 'INCOMP::MEG[0.5]'
superheat = 4.0  # K
p_SL = 1e5  # Pa

k_cond = [380., 800., 380.]
area_cond = 0.565

k_evap = [500., 100.]
area_evap = 1.

# boundary conditions
T_SL_cold_in = -10.0 + 273.15  # K
T_SL_hot_in = 30.0 + 273.15  # K
h_SL_cold_in = CPPSI('H', 'T', T_SL_cold_in, 'P', p_SL, SL)
h_SL_hot_in = CPPSI('H', 'T', T_SL_hot_in, 'P', p_SL, SL)
mdot_SL_cold = 0.1  # kg/s
mdot_SL_hot = 0.1  # kg/s

# initial guesses
mdot_ref_init = .5e-3
pc_init = 13e5
Tc_init = CPPSI('T', 'P', pc_init, 'Q', 0, ref)
h2_init = CPPSI('H', 'P', pc_init, 'T', 60+273.15, ref)
h3_init = CPPSI('H', 'P', pc_init, 'Q', 0, ref)
h4_init = CPPSI('H', 'P', pc_init, 'T', Tc_init-2., ref)
p0_init = 3.2e5
T0_init = CPPSI('T', 'P', p0_init, 'Q', 1, ref)
h5_init = CPPSI('H', 'P', p0_init, 'T', T0_init+superheat, ref)
h1_init = CPPSI('H', 'P', p0_init, 'T', T0_init+superheat+2., ref)

initial_areafraction_cond = [0.2, 0.6, 0.2]
initial_areafraction_evap = [0.2, 0.8]

# Instantiate components
system = vcs.System(id='system', tolerance=1.)
cpr = vcs.CompressorEfficiency(id='cpr', system=system, etaS=0.645, etaV=0.82, etaEL=0.775, stroke=cpr_stroke, speed=cpr_speed)
cond = vcs.CondenserBPHE(id='cond', system=system, k=k_cond, area=area_cond, subcooling=0.1, initial_areafractions=initial_areafraction_cond)
ihx = vcs.IHX(id='ihx', system=system, UA=2.3)
evap = vcs.Evaporator(id='evap', system=system, k=k_evap, area=area_evap, superheat=superheat, boundary_switch=True, limit_temp=True, initial_areafractions=initial_areafraction_evap)

snkcold = vcs.Sink(id='snkcold', system=system)
srccold = vcs.Source(id='srccold', system=system, mdot=mdot_SL_cold, p=p_SL, h=h_SL_cold_in)

snkhot = vcs.Sink(id='snkhot', system=system)
srchot = vcs.Source(id='srchot', system=system, mdot=mdot_SL_hot, p=p_SL, h=h_SL_hot_in)

# connections
cpr_cond = vcs.Junction(id='cpr_cond', system=system, medium=ref, upstream_component=cpr, upstream_id='outlet_A', downstream_component=cond, downstream_id='inlet_A', mdot_init=mdot_ref_init, p_init=pc_init, h_init=h2_init)
cond_ihx = vcs.Junction(id='cond_ihx', system=system, medium=ref, upstream_component=cond, upstream_id='outlet_A', downstream_component=ihx, downstream_id='inlet_A', mdot_init= mdot_ref_init, p_init=pc_init, h_init=h3_init)
ihx_evap = vcs.Junction(id='ihx_evap', system=system, medium=ref, upstream_component=ihx, upstream_id='outlet_A', downstream_component=evap, downstream_id='inlet_A', mdot_init= mdot_ref_init, p_init=p0_init, h_init=h4_init)
evap_ihx = vcs.Junction(id='evap_ihx', system=system, medium=ref, upstream_component=evap, upstream_id='outlet_A', downstream_component=ihx, downstream_id='inlet_B', mdot_init=mdot_ref_init, p_init=p0_init, h_init=h5_init)
ihx_cpr = vcs.Junction(id='ihx_cpr', system=system, medium=ref, upstream_component=ihx, upstream_id='outlet_B', downstream_component=cpr, downstream_id='inlet_A', mdot_init=mdot_ref_init, p_init=p0_init, h_init=h1_init)

srccold_evap = vcs.Junction(id='srccold_evap', system=system, medium=SL, upstream_component=srccold, upstream_id='outlet_A', downstream_component=evap, downstream_id='inlet_B', mdot_init=mdot_SL_cold, p_init=p_SL, h_init=h_SL_cold_in)
evap_snkcold = vcs.Junction(id='evap_snkcold', system=system, medium=SL, upstream_component=evap, upstream_id='outlet_B', downstream_component=snkcold, downstream_id='inlet_A', mdot_init=mdot_SL_cold, p_init=p_SL, h_init=h_SL_cold_in)

srchot_cond = vcs.Junction(id='srchot_cond', system=system, medium=SL, upstream_component=srchot, upstream_id='outlet_A', downstream_component=cond, downstream_id='inlet_B', mdot_init=mdot_SL_hot, p_init=p_SL, h_init=h_SL_hot_in)
cond_snkhot = vcs.Junction(id='cond_snkhot', system=system, medium=SL, upstream_component=cond, upstream_id='outlet_B', downstream_component=snkhot, downstream_id='inlet_A', mdot_init=mdot_SL_hot, p_init=p_SL, h_init=h_SL_hot_in)

system.initialize()
system.run(full_output=True)

Q0 = (cond_ihx.get_enthalpy() - evap_ihx.get_enthalpy()) * cond_ihx.get_massflow()
QC = (cpr_cond.get_enthalpy() - cond_ihx.get_enthalpy()) * cpr_cond.get_massflow()
Pel = cpr.get_power()

print('Q0: \t{:.2f} W'.format(Q0))
print('QC: \t{:.2f} W'.format(QC))
print('Pel: \t{:.2f} W'.format(Pel))

x_evap_in = CPPSI('Q', 'H', ihx_evap.get_enthalpy(), 'P', evap.p, ref)

plot_dict = {
    'cycle': {
        'junctions':
            {
                'ihx_cpr':  {'component': ihx_cpr,  'position': [635, 215], 'ha': 'left', 'va': 'top', 'ref': True},
                'cpr_cond': {'component': cpr_cond, 'position': [635, 50], 'ha': 'left', 'va': 'top', 'ref': True},
                'cond_ihx': {'component': cond_ihx, 'position': [70, 20], 'ha': 'left', 'va': 'bottom', 'ref': True},
                'ihx_evap': {'component': ihx_evap, 'position': [155, 320], 'ha': 'left', 'va': 'top', 'ref': True},
                'evap_ixh': {'component': evap_ihx, 'position': [635, 400], 'ha': 'left', 'va': 'top', 'ref': True},
                'src_evap': {'component': srccold_evap, 'position': [420, 640], 'ha': 'left', 'va': 'top'},
                'evap_snk': {'component': evap_snkcold, 'position': [260, 640], 'ha': 'right', 'va': 'top'}
            },
        'special':
            {
                'txv_evap': {'position': [80, 535], 'ha': 'left', 'va': 'top','text': 'T: {T:.2f} °C\np: {p:.2f} bar\nh:{h:.2f} J/kg\nmdot: {mdot:.2f} g/s\nx: {x:0.3f}'.format(T=evap.T0-273.15, p=evap.p*1e-5, h=ihx_evap.h, mdot=ihx_evap.mdot*1e3, x=x_evap_in)},
                'general': {'position': [240, 100], 'ha':'left', 'va': 'top', 'text': 'T_box: {T_box: .2f} °C\nT_amb: {T_amb: .2f} °C\nPel: {Pel: .2f} W\nQ_dot: {Qdot: .2f}\nn_CPR: {n_CPR: .0f} rpm\nVdot_box: {Vbox:.0f} m3/h\nVdot_amb: {Vamb:.0f} m3/h'.format(Pel=cpr.Pel, Qdot=Q0, n_CPR= cpr_speed, T_box=T_SL_cold_in-273.15, T_amb=T_SL_hot_in-273.15, Vbox=mdot_SL_cold*3600, Vamb=mdot_SL_hot*3600)}
            }
    },
    'refrigerant': ref
}
system.plot_cycle(dict=plot_dict, cycle_img_path=r'diagram.png')
