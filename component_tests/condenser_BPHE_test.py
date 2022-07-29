import VCSpbx as vcs
from CoolProp.CoolProp import PropsSI as CPPSI

# Testing the CondenserBPHE class

SL = 'INCOMP::MEG[0.5]'
ref = 'R290'
T_SL_in = 30. + 273.15
h_SL_in = CPPSI('H', 'T', T_SL_in, 'P', 1e5, SL)
mdot_ref = 0.01
p_ref = 14e5
h_ref_in = CPPSI('H', 'T', 80+273.15, 'P', p_ref, ref)
mdot_SL = 0.5
p_SL = 1e5

# components
system = vcs.System(id='system', tolerance=1e-4)
cond = vcs.CondenserBPHE(id='cond', system=system, k=[300., 300., 300.], area=0.8, subcooling=0.1)
src_SL = vcs.Source(id='src_SL', system=system, mdot=mdot_SL, p=p_SL, h=h_SL_in)
snk_SL = vcs.Sink('snk_SL', system)
src_ref = vcs.Source('src_ref', system, mdot_ref, p_ref, h_ref_in)
snk_ref = vcs.Sink('snk_ref', system)

# junctions
srcref_cond = vcs.Junction('srcref_cond', system, ref, src_ref, 'outlet_A', cond, 'inlet_A', mdot_ref, p_ref, h_ref_in)
cond_snkref = vcs.Junction('cond_snkref', system, ref, cond, 'outlet_A', snk_ref, 'inlet_A', mdot_ref, p_ref, h_ref_in)
srcSL_cond = vcs.Junction('srcSL_cond', system, SL, src_SL, 'outlet_A', cond, 'inlet_B', mdot_SL, p_SL, h_SL_in)
cond_snkSL = vcs.Junction('cond_snkSL', system, SL, cond, 'outlet_B', snk_SL, 'inlet_A', mdot_SL, p_SL, h_SL_in)

system.run(full_output=True)

cond.calc()
