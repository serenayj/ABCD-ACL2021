import sys 
import pickle 
import re 
from nltk import sent_tokenize 

from io import StringIO
import networkx as nx 

from data import *
from utils import * 
from distant_superv import *

batch_id = "test" 
data_path = "data"

sents = open(data_path+"/"+str(batch_id)+".complex").readlines()
orig_sents = [s.strip() for s in sents] 

deplines = open(data_path+"/"+str(batch_id)+".complex.out").readlines() 
locs = [ind for ind, value in enumerate(deplines) if "Dependency Parse (enhanced plus plus dependencies):\n" in value]
sent_locs = [ind for ind, value in enumerate(deplines) if "Sentence #" in value]

output_arcs = {} 
for _ in range(0, len(sents)):
	loc = locs[_]
	if _ == len(sents) -1:
		dep_lines = deplines[loc:]
		# elif _ == 1073:
		# 	dep_lines = deplines[sent_locs[_]:]
	else: 
		dep_lines = deplines[loc+1:sent_locs[_+1]] 

	special = ['-LRB-', "-RRB-"]
	tups = [] 
	words = [] 
	for t in dep_lines:
		if t == "\n":
			"""
			If it is the end of dependency parse, break  
			"""
			break 
		if t == "Dependency Parse (enhanced plus plus dependencies):\n":
			"""
			Starts from the beginning of dependency parsing 
			"""
			continue
		arc = t.split("(")[0]
		#print(arc)
		if arc =="punct" or arc=="root":
			continue 
		tokens = t.split("(")[-1].split(")")[:-1][0].split(",")
		
		if len(tokens) >2:
			src = tokens[0].strip()
			tgt = ','+tokens[-1].strip()
		else:
			src = tokens[0].strip()
			tgt = tokens[1].strip() 
			if "-LRB-" in src:
				#print("src ", src) 
				src = src.replace("-LRB-", "LRB") 
			if "-RRB-" in src:
				#print("src ", src) 
				src = src.replace("-RRB-", "RRB") 
			if "-RRB-" in tgt:
				#print("tgt ", tgt) 
				tgt = tgt.replace("-RRB-", "RRB") 
			if "-LRB-" in tgt:
				#print("tgt ", tgt) 
				tgt = tgt.replace("-LRB-", "LRB") 
		for i in [src, tgt]:
			if i not in words and i != '':
				i = i.split("-")[0] + "-"+re.sub("[^0-9]", "", i.split("-")[1])
				#print(i)
				if i != "-":
					words.append(i) 
		if src == '' or tgt == '':
			continue 
		if src  == '_' or tgt == "_":
			continue 
		else:
			newsrc = src.split("-")[0] + "-"+re.sub("[^0-9]", "", src.split("-")[1])
			newtgt = tgt.split("-")[0] + "-"+re.sub("[^0-9]", "", tgt.split("-")[1])
			tups.append((newsrc, newtgt, arc))

	output_arcs[_] = {}
	vtov, vtoi = wordict(words) 
	itov = {v:k for k,v in vtoi.items()}
	adjs, conn_adjs = BuildAdj(vtoi, tups)
	#adjs, conn_adjs = BuildAdj(vtoi, tmp) # connected nodes by neighbor, adj: nodes connected by dependency arcs
	# Get all pairs of src, tgt, arcs 
	all_pairs = BuildPairs(adjs) 
	# Add self arc to indicate if a word being dropped  
	all_pairs = AddSelf(itov, all_pairs) 

	output_arcs[_]["orig"] = sents[_].strip()
	output_arcs[_]['adjs'], output_arcs[_]['all_pairs'] = adjs, all_pairs 
	output_arcs[_]['words'] = words 
	output_arcs[_]['vtoi'], output_arcs[_]['itov'] = vtoi, itov 


with open(data_path+'/batch_'+str(batch_id)+'.pkl', 'wb') as handle:
	pickle.dump(output_arcs, handle, protocol=pickle.HIGHEST_PROTOCOL)

