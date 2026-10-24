from Rockets.Simulation.Plots import from_md, ColorMappingMD


mapping = from_md(ColorMappingMD)

mapping['Fn'] = mapping.F.str[:1].astype(int)
mapping['Bn'] = mapping.B.str[:1].astype(int)
mapping['En'] = mapping.E.str[:1].astype(int)

mapping['check'] = mapping.Fn*1 + mapping.Bn*4 + mapping.En*2 

assert all(mapping.index == mapping.check)