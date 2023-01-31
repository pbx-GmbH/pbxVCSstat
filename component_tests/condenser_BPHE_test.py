import numpy as np
import VCSpbx as vcs
from CoolProp.CoolProp import PropsSI as CPPSI

# Testing the CondenserBPHE class

SL = 'INCOMP::MEG[0.5]'  # coolant
ref = 'R290'  # refrigerant
T_SL_in = 30. + 273.15  # inlet temperature coolant
h_SL_in = CPPSI('H', 'T', T_SL_in, 'P', 1e5, SL)  # inlet enthalpy coolant
mdot_ref = 0.01  # mass flow refrigerant
p_ref = 14e5  # pressure guess refrigerant
h_ref_in = CPPSI('H', 'T', 80+273.15, 'P', p_ref, ref)  # inlet enthalpy refrigerant
mdot_SL = 0.5  # mass flow coolant
p_SL = 1e5  # pressure coolant

# components
system = vcs.System(id='system', tolerance=1e-8)
cond = vcs.CondenserBPHE(id='cond', system=system, k=[300., 3000., 300.], area=0.8, subcooling=0.1)
srcSL = vcs.Source(id='src_SL', system=system, mdot=mdot_SL, p=p_SL, h=h_SL_in)
snkSL = vcs.Sink('snk_SL', system)
srcRef = vcs.Source('src_ref', system, mdot_ref, p_ref, h_ref_in)
snkRef = vcs.Sink('snk_ref', system)

# junctions
srcRef_cond = vcs.Junction('srcref_cond', system, ref, srcRef, 'outlet_A', cond, 'inlet_A', mdot_ref, p_ref, h_ref_in)
cond_snkRef = vcs.Junction('cond_snkref', system, ref, cond, 'outlet_A', snkRef, 'inlet_A', mdot_ref, p_ref, h_ref_in)
srcSL_cond = vcs.Junction('srcSL_cond', system, SL, srcSL, 'outlet_A', cond, 'inlet_B', mdot_SL, p_SL, h_SL_in)
cond_snkSL = vcs.Junction('cond_snkSL', system, SL, cond, 'outlet_B', snkSL, 'inlet_A', mdot_SL, p_SL, h_SL_in)


# system.run(full_output=True)
#
# system.initialize()
# cond.calc()
#
# Qdot_refside = (srcRef_cond.get_enthalpy() - cond_snkRef.get_enthalpy()) * srcRef_cond.get_massflow()
# Qdot_coolantside = (srcSL_cond.get_enthalpy() - cond_snkSL.get_enthalpy()) * srcSL_cond.get_massflow()
# print('Energy balance:')
# print('Qdot_refside = {}'.format(Qdot_refside))
# print('Qdot_coolantside = {}'.format(Qdot_coolantside))
# print('Difference = {}'.format(Qdot_refside+Qdot_coolantside))

mdot_ref_range = np.arange(0.01, 0.05, 0.005)
Qdot_refside = np.array([])
Qdot_coolantside = np.array([])
pC = np.array([])

print('Relative error:')
for mdot_ref_set in mdot_ref_range:
    srcRef.set_mdot(mdot_ref_set)
    system.run()
    Qdot_refside = cond.mdot_ref * (srcRef_cond.h - cond_snkRef.h)
    Qdot_coolantside = cond.mdot_SL * (srcSL_cond.h - cond_snkSL.h)
    print((Qdot_refside+Qdot_coolantside)/Qdot_coolantside)

system.get_export_variables()

