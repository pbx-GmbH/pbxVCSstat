import VCSpbx as vcs
from CoolProp.CoolProp import PropsSI as CPPSI

# parameter setting
cpr_speed = 4000.0  # rpm
cpr_stroke = 33.0e-6
ref = 'R290'
SL = 'INCOMP::MEG[0.5]'
superheat = 6.0  # K
p_SL = 1e5  # Pa

k_cond = [380., 380., 380.]
area_cond = 0.565

k_evap = [420., 420.]
area_evap = 1.

# boundary conditions
T_SL_cold_in = -1.0 + 273.15
T_SL_hot_in = 35.0 + 273.15
h_SL_cold_in = CPPSI('H', 'T', T_SL_cold_in, 'P', p_SL, SL)
h_SL_hot_in = CPPSI('H', 'T', T_SL_hot_in, 'P', p_SL, SL)
mdot_SL_cold = 0.15  # kg/s
mdot_SL_hot = 0.15  # kg/s

# initial guesses
mdot_ref_init = 5.e-3
pc_init = 18e5
Tc_init = CPPSI('T', 'P', pc_init, 'Q', 0, ref)
h2_init = CPPSI('H', 'P', pc_init, 'T', 60+273.15, ref)
h3_init = CPPSI('H', 'P', pc_init, 'Q', 0, ref)
h4_init = CPPSI('H', 'P', pc_init, 'T', Tc_init-2., ref)
p0_init = 3.0e5
T0_init = CPPSI('T', 'P', p0_init, 'Q', 1, ref)
h5_init = CPPSI('H', 'P', p0_init, 'T', T0_init+superheat, ref)
h1_init = CPPSI('H', 'P', p0_init, 'T', T0_init+superheat+2., ref)

initial_areafraction_cond = [0.2, 0.6, 0.2]
initial_areafraction_evap = [0.2, 0.8]

# Instantiate components
system = vcs.System(id='system', tolerance=1.)
cpr = vcs.CompressorEfficiency(id='cpr', system=system, etaS=0.645, etaV=0.82, etaEL=0.775, stroke=15e-6, speed=cpr_speed)
cond = vcs.CondenserBPHE(id='cond', system=system, k=k_cond, area=area_cond, subcooling=0.1, initial_areafractions=initial_areafraction_cond)
ihx = vcs.IHX(id='ihx', system=system, UA=5.3)
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

system.run(full_output=True)
