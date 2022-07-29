import VCSpbx as vcs
from CoolProp.CoolProp import PropsSI as CPPSI

# parameter setting
cpr_speed = 4000.0  # rpm
T_amb = 50. + 273.15  # K
T_box = 4. + 273.15  # K
ref = 'R290'
air = 'AIR'
mdot_air = 0.35  # kg/s
p_air = 1e5  # Pa
superheat = 6.0  # K
mdot_air_cond = 0.3  # kg/s

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

h_air_in = CPPSI('H', 'T', T_box, 'P', p_air, air)

# Instantiate components
system = vcs.System(id='system', tolerance=1.)
cpr = vcs.Compressor_efficiency(id='cpr', system=system, etaS=0.645, etaV=0.82, etaEL=0.775, stroke=15e-6, speed=cpr_speed)
cond = vcs.Condenser(id='cond', system=system, k=[380., 380., 380.], area=0.565, subcooling=0.1, T_air_in=T_amb, mdot_air_in=mdot_air_cond)
ihx = vcs.IHX(id='ihx', system=system, UA=5.3)
evap = vcs.Evaporator(id='evap', system=system, k=[420., 420.], area=1., superheat=superheat, boundary_switch=True, limit_temp=True)
snkair = vcs.Sink(id='snk_air', system=system)
srcair = vcs.Source(id='src_air', system=system, mdot=mdot_air, p=p_air, h=h_air_in)

# connections
cpr_cond = vcs.Junction(id='cpr_cond', system=system, medium=ref, upstream_component=cpr, upstream_id='outlet_A', downstream_component=cond, downstream_id='inlet_A', mdot_init=mdot_ref_init, p_init=pc_init, h_init=h2_init)
cond_ihx = vcs.Junction(id='cond_ihx', system=system, medium=ref, upstream_component=cond, upstream_id='outlet_A', downstream_component=ihx, downstream_id='inlet_A', mdot_init= mdot_ref_init, p_init=pc_init, h_init=h3_init)
ihx_evap = vcs.Junction(id='ihx_evap', system=system, medium=ref, upstream_component=ihx, upstream_id='outlet_A', downstream_component=evap, downstream_id='inlet_A', mdot_init= mdot_ref_init, p_init=p0_init, h_init=h4_init)
evap_ihx = vcs.Junction(id='evap_ihx', system=system, medium=ref, upstream_component=evap, upstream_id='outlet_A', downstream_component=ihx, downstream_id='inlet_B', mdot_init=mdot_ref_init, p_init=p0_init, h_init=h5_init)
ihx_cpr = vcs.Junction(id='ihx_cpr', system=system, medium=ref, upstream_component=ihx, upstream_id='outlet_B', downstream_component=cpr, downstream_id='inlet_A', mdot_init=mdot_ref_init, p_init=p0_init, h_init=h1_init)
srcair_evap = vcs.Junction(id='srcair_evap', system=system, medium=air, upstream_component=srcair, upstream_id='outlet_A', downstream_component=evap, downstream_id='inlet_B', mdot_init=mdot_air, p_init=p_air, h_init=h_air_in)
evap_snkair = vcs.Junction(id='evap_snkair', system=system, medium=air, upstream_component=evap, upstream_id='outlet_B', downstream_component=snkair, downstream_id='inlet_A', mdot_init=mdot_air, p_init=p_air, h_init=h_air_in)

system.run(full_output=True)

Qdot = evap_ihx.mdot*(evap_ihx.h - ihx_evap.h)
x_evap_in = CPPSI('Q', 'H', ihx_evap.get_enthalpy(), 'P', evap.p, ref)
Vdot_box = mdot_air * CPPSI('D', 'T', T_box, 'P', p_air, 'AIR')
Vdot_air_cond = mdot_air_cond * CPPSI('D', 'T', T_amb, 'P', p_air, 'AIR')
plot_dict = {
    'cycle': {
        'junctions':
            {
                'ihx_cpr':  {'component': ihx_cpr,  'position': [635, 215], 'ha': 'left', 'va': 'top', 'ref': True},
                'cpr_cond': {'component': cpr_cond, 'position': [635, 50], 'ha': 'left', 'va': 'top', 'ref': True},
                'cond_ihx': {'component': cond_ihx, 'position': [70, 20], 'ha': 'left', 'va': 'bottom', 'ref': True},
                'ihx_evap': {'component': ihx_evap, 'position': [155, 320], 'ha': 'left', 'va': 'top', 'ref': True},
                'evap_ixh': {'component': evap_ihx, 'position': [635, 400], 'ha': 'left', 'va': 'top', 'ref': True},
                'src_evap': {'component': srcair_evap, 'position': [420, 640], 'ha': 'left', 'va': 'top'},
                'evap_snk': {'component': evap_snkair, 'position': [260, 640], 'ha': 'right', 'va': 'top'}
            },
        'special':
            {
                'txv_evap': {'position': [80, 535], 'ha': 'left', 'va': 'top','text': 'T: {T:.2f} °C\np: {p:.2f} bar\nh:{h:.2f} J/kg\nmdot: {mdot:.2f} g/s\nx: {x:0.3f}'.format(T=evap.T0-273.15, p=evap.p*1e-5, h=ihx_evap.h, mdot=ihx_evap.mdot*1e3, x=x_evap_in)},
                'general': {'position': [240, 100], 'ha':'left', 'va': 'top', 'text': 'T_box: {T_box: .2f} °C\nT_amb: {T_amb: .2f} °C\nPel: {Pel: .2f} W\nQ_dot: {Qdot: .2f}\nn_CPR: {n_CPR: .0f} rpm\nVdot_box: {Vbox:.0f} m3/h\nVdot_amb: {Vamb:.0f} m3/h'.format(Pel=cpr.Pel, Qdot=Qdot, n_CPR= cpr_speed, T_box=T_box-273.15, T_amb=T_amb-273.15, Vbox=Vdot_box*3600, Vamb=Vdot_air_cond*3600)}
            }
    },
    'refrigerant': ref
}
system.plotCycle(dict=plot_dict, cycle_img_path=r'diagram.png')
