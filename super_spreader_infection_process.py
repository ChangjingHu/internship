import matplotlib.pyplot as plt
import numpy as np
import dynetx as dn
import pandas as pd
import networkx as nx
import csv
import numpy as np



G = nx.Graph()
#construct a graph in order to record the edges between nodes
all_population = [i for i in range(1, 387583)]
#print(all_population)
G.add_nodes_from(all_population)
#all candidates can be considered as isolated nodes
data = pd.read_csv(r"C:\Users\CharlotteHoo\Desktop\internship\data_to_2020_3_8\results.csv", header=None)
data.columns = ["timestamp", "location_ids", "infector_ids", "infected_ids", "infection_ids", "location_spaces", "region_names"]

#find the ids of super_spreader
data_1 = pd.read_csv(r"C:\Users\CharlotteHoo\Desktop\internship\data_to_2020_3_8\super_spreaders_dictionary.csv", header=None)
data_1.columns = ["super_spreader_id","number of infected people"]
super_spread_ids = data_1["super_spreader_id"].values.tolist()
#organize data table
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
g.add_nodes_from(seed_infector)
#start a dynamic network which can change networks over time, and time step is one day here

merges = []

for i in range(len(dates)):
	subgroup = group.get_group(dates[i])
#acquire the data of a certain day in chronological order
	G = nx.from_pandas_edgelist(subgroup, "infector_ids", "infected_ids")
	#add edges to the network graph used to record new rounds of infections, and G is an intermediate container
	#add new edges for each time step
	g.add_interactions_from(G.edges(), t=i)
	#add the newly added edges in G at time t to the dynamic graph g
	super_spread_status={}
	for s in super_spread_ids:
		if s in g.nodes():
			if s in seed_infector:
				update_dictionary = {}
				update_dictionary[s] = g.degree(s)
				super_spread_status.update(update_dictionary)
			else:
				update_dictionary = {}
				update_dictionary[s] = g.degree(s)-1
				super_spread_status.update(update_dictionary)
		else:
			update_dictionary = {}
			update_dictionary[s] = 0
			super_spread_status.update(update_dictionary)
	merges.append(super_spread_status)
	print("when ", dates[i])
	print("Number of people infected by super-spreaders that day ",super_spread_status)
	print("--------------------------------------------------------------------------------------")

masterdf = pd.DataFrame()
colnames = ['2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09', '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13', '2020-03-14', '2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18', '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31', '2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04', '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-08', '2020-04-09', '2020-04-10', '2020-04-11', '2020-04-12', '2020-04-13', '2020-04-14', '2020-04-15', '2020-04-16', '2020-04-17', '2020-04-18', '2020-04-19', '2020-04-20', '2020-04-21', '2020-04-22', '2020-04-23', '2020-04-24', '2020-04-25', '2020-04-26', '2020-04-27', '2020-04-28', '2020-04-29', '2020-04-30', '2020-05-01', '2020-05-02', '2020-05-03', '2020-05-04', '2020-05-05', '2020-05-06', '2020-05-07', '2020-05-08', '2020-05-09', '2020-05-10', '2020-05-11', '2020-05-12', '2020-05-13', '2020-05-14', '2020-05-15', '2020-05-16', '2020-05-17', '2020-05-18', '2020-05-19', '2020-05-20', '2020-05-21', '2020-05-22', '2020-05-23', '2020-05-24', '2020-05-25', '2020-05-26', '2020-05-27', '2020-05-28', '2020-05-29', '2020-05-30', '2020-05-31', '2020-06-01', '2020-06-02', '2020-06-03', '2020-06-04', '2020-06-05', '2020-06-06']
for i in merges:
	df = pd.DataFrame([i]).T
	masterdf = pd.concat([masterdf, df], axis=1)

# assign the column names
masterdf.columns = colnames

# get a glimpse of what the data frame looks like
masterdf.head()

# save to csv
masterdf.to_csv("C:\\Users\\CharlotteHoo\\Desktop\\internship\\data_to_2020_3_8\\super_merge.csv", index=True)








