
from dingo.core.network import *
from dingo.core.structure import *
from oemof.core import energy_system as es

#bel = Bus(uid="electricity", type="electricity")
bus0 = BusDingo(uid="b0", type="electricity")
bus1 = BusDingo(uid="b1", type="electricity")
bus2 = BusDingo(uid="b2", type="electricity")
bus3 = BusDingo(uid="b3", type="electricity")
bus4 = BusDingo(uid="b4", type="electricity")
bus5 = BusDingo(uid="b5", type="electricity")
bus6 = BusDingo(uid="b6", type="electricity")
bus7 = BusDingo(uid="b7", type="electricity")
bus8 = BusDingo(uid="b8", type="electricity")
bus9 = BusDingo(uid="b9", type="electricity")
bus10 = BusDingo(uid="b10", type="electricity")

#x = Line(uid="bla", inputs=[b1], outputs=[b2])
line1 = BranchDingo(uid="l1", inputs=[bus1], outputs=[bus2])
line2 = BranchDingo(uid="l2", inputs=[bus2], outputs=[bus3])
line3 = BranchDingo(uid="l3", inputs=[bus3], outputs=[bus4])
line4 = BranchDingo(uid="l4", inputs=[bus5], outputs=[bus6])
line5 = BranchDingo(uid="l5", inputs=[bus5], outputs=[bus7])
line6 = BranchDingo(uid="l6", inputs=[bus7], outputs=[bus8])
trans1 = TransformerDingo(uid="t1", inputs=[bus0], outputs=[bus1])
trans2 = TransformerDingo(uid="t2", inputs=[bus2], outputs=[bus5])
trans3 = TransformerDingo(uid="t3", inputs=[bus3], outputs=[bus9])
trans4 = TransformerDingo(uid="t4", inputs=[bus4], outputs=[bus10])

#grouping

buses = [bus0, bus1, bus2, bus3, bus4, bus5, bus6, bus7, bus8, bus9, bus10]
lines = [line1, line2, line3, line4, line5, line6]
transformators = [trans1, trans2, trans3, trans4]
transports =  lines + transformators

# group components
#transformers = [pp_coal, pp_lig, pp_gas, pp_oil, pp_chp]
#renew_sources = [pv, wind_on]
#sinks = [demand_th, demand_el]

#components = transformers + renew_sources + sinks
components = transports
entities = components + buses

#simulation = es.Simulation(
#    solver='glpk', timesteps=timesteps, stream_solver_output=True,
#    objective_options={'function': predefined_objectives.minimize_cost})
energysystem = es.EnergySystem(entities=entities)

#energysystem.plot_as_graph(labels=1)
