import matplotlib.pyplot as plt
import numpy as np
import dynetx as dn
import pandas as pd
import networkx as nx
import csv
import numpy as np
from matplotlib import animation


data = pd.read_csv(r"C:\Users\CharlotteHoo\Desktop\internship\data_to_2020_3_8\results.csv", header=None)
data.columns = ["timestamp", "location_ids", "infector_ids", "infected_ids", "infection_ids", "location_spaces", "region_names"]
data.drop(data.index[data['infector_ids'] == data['infected_ids']], inplace=True)
#find the ids of super_spreader
data_1 = pd.read_csv(r"C:\Users\CharlotteHoo\Desktop\internship\data_to_2020_3_8\super_spreaders_dictionary.csv", header=None)
data_1.columns = ["super_spreader_id","number of infected people"]
super_spread_ids = data_1["super_spreader_id"].values.tolist()
#organize data table

G = nx.Graph()
#construct a graph in order to record the edges between nodes
G.add_nodes_from(super_spread_ids)


group = data.groupby('timestamp')
#divide infection events by date
seed_infector = data[(data["location_spaces"] == "infection_seed")].infected_ids
seed_infector = seed_infector.values.tolist()

dates = []
#list for time series
for key, value in group:
	dates.append(key)
#here "key" is date, and date is put in dates[]

g = dn.DynGraph(edge_removal=True)
#g.add_nodes_from(seed_infector)
#start a dynamic network which can change networks over time, and time step is one day here

merges = 0

for i in range(len(dates)):
	subgroup = group.get_group(dates[i])
	#print(list(subgroup.loc[:,"infector_ids"]))
	#acquire the data of a certain day in chronological order
	for item in list(subgroup.loc[:,"infector_ids"]):
		if not item in super_spread_ids:
			subgroup = subgroup[subgroup.infector_ids != item]
	G = nx.from_pandas_edgelist(subgroup, "infector_ids", "infected_ids")
	#add edges to the network graph used to record new rounds of infections, and G is an intermediate container
	#add new edges for each time step
	for v in list(G):
		if G.degree(v) == 0:
			G.remove_node(v)
	g.add_interactions_from(G.edges(), t=i)
	#add the newly added edges in G at time t to the dynamic graph g
	G = nx.erdos_renyi_graph(20, 0.1)
	color_map = []
	colors = [n for n in range(len(g.degree()))]
	de = g.degree()
	degrees = list(de.values())
	for i in range(len(de)):
		colors[i] = degrees[i]/60
	nx.draw(g,pos=None, with_labels=False, cmap=plt.get_cmap('Reds'), node_color=colors, width=0.05, node_size=30)
	plt.pause(0.5)


