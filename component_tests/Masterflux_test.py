import VCSpbx as vcs
from CoolProp.CoolProp import PropsSI as CPPSI

# parameters
cpr_speed = 4000.  # rpm
suction_pressure = 3.0e5  # Pa
discharge_pressure = 15e5  # Pa
superheat = 10.  # K
ref = 'R290'  # refrigerant
t0 = CPPSI('T', 'P', suction_pressure, 'Q', 0, ref)

# initial guesses
t_discharge_0 = 70. + 273.15  # K
mdot_init = 5.e-3  # kg/s
h_out_0 = CPPSI('H', 'T', t_discharge_0, 'P', discharge_pressure, ref)
h_in_0 = CPPSI('H', 'T', t0 + superheat, 'P', suction_pressure, ref)

# system and components
system = vcs.System(id='system', tolerance=1.0E-4)
cpr = vcs.Compressor_MasterfluxAlpine('cpr', system, cpr_speed)
source = vcs.Source('source', system, mdot_init, suction_pressure, h_in_0)
sink = vcs.Sink('sink', system, mdot_init, discharge_pressure, h_out_0)

# connections
src_cpr = vcs.Junction('src_cpr', system, ref, source, 'outlet_A', cpr, 'inlet_A', mdot_init, suction_pressure, h_in_0)
cpr_snk = vcs.Junction('cpr_snk', system, ref, cpr, 'outlet_A', sink, 'inlet_A', mdot_init, discharge_pressure, h_out_0)

system.initialize()
system.run(full_output=True)
print('Pel: {:2f}'.format(cpr.Pel))
