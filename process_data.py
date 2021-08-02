"""
debug process pipeline 
"""

import sys 
import pickle 
import re 
from nltk import sent_tokenize 

from io import StringIO
import networkx as nx 

from data import *
from utils import * 
from distant_superv import *

#import stanfordnlp
#nlp = stanfordnlp.Pipeline()



"""
if stanford nlp 
"""

#batch_id = sys.argv[1] 
batch_id = 0 
def GetDeps(dep_lines):
	special = ['-LRB-', "-RRB-"]
	tups = [] 
	words = [] 
	try:
		for t in dep_lines:
			arc = t.split("(")[0]
			tokens = t.split("(")[-1].split(")")[:-1][0].split(",")
			if arc =="punct" or arc=="root":
				continue 

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
				#print("new src", src)
				#print("new tgt", tgt)
				newsrc = src.split("-")[0] + "-"+re.sub("[^0-9]", "", src.split("-")[1])
				newtgt = tgt.split("-")[0] + "-"+re.sub("[^0-9]", "", tgt.split("-")[1])
				tups.append((newsrc, newtgt, arc))
	except:
		return False, False 
	return tups, words 

sents = open("data/clean_orig_sent_"+str(batch_id)+".txt").readlines()
orig_sents = [s.strip() for s in sents] 

golds = open("data/clean_gold_sent_"+str(batch_id)+".txt").readlines()
gold_sents = [s.strip() for s in golds]

deplines = open("data/clean_orig_sent_"+str(batch_id)+".txt.out").readlines() 
locs = [ind for ind, value in enumerate(deplines) if "Dependency Parse (enhanced plus plus dependencies):\n" in value]
sent_locs = [ind for ind, value in enumerate(deplines) if "Sentence #" in value]

output_arcs = {} 
#for _ in range(0, len(sents)):
for _ in range(4734,4734+1):
	loc = locs[_]
	if _ == len(sents) -1:
		dep_lines = deplines[loc:]
	else: 
		dep_lines = deplines[loc+1:sent_locs[_+1]] 

	tmp, words = GetDeps(dep_lines)
	if tmp == False and words ==False:
		print("Skipping problematic sentence because of token errors", _)
		continue 
	try:
		output_arcs[_] = {}
		vtov, vtoi = wordict(words) 
		itov = {v:k for k,v in vtoi.items()}
		adjs, conn_adjs = BuildAdj(vtoi, tmp) # connected nodes by neighbor, adj: nodes connected by dependency arcs
		# Get all pairs of src, tgt, arcs 
		all_pairs = BuildPairs(adjs) 
		# Add self arc to indicate if a word being dropped  
		all_pairs = AddSelf(itov, all_pairs) 
		
		#tmp = "Sokuhi was born in Fuzhou , Fujian , China. Sokuhi was ordained at 17 by Feiyin Tongrong."
		g_temp = gold_sents[_]
		
		golds_out = sent_tokenize(g_temp)
		golds_out = [i.replace("-LRB-","LRB").replace("-RRB-","RRB") for i in golds_out]
		s = orig_sents[_].replace("-LRB-","LRB").replace("-RRB-","RRB")
		where_it_from, new_temp, cut_temp, del_temp =  KEEPORDROP(all_pairs, golds_out, itov, s) 
		#new_temp, cut_temp, del_temp =  KEEP(all_pairs, golds, itov, s) 
		copy_temp = COPY(all_pairs, golds, itov,where_it_from)
	except:
		print("Skipping problematic sentence because of token errors", _)
		continue 

	if len(new_temp) == 0 :
		print("skipping sentence at index ", _ )
		continue 
	else:
		output_arcs[_]['accept'], output_arcs[_]['copy'], output_arcs[_]['break'], output_arcs[_]['drop'] =  new_temp, copy_temp, cut_temp, del_temp 
		output_arcs[_]['adjs'], output_arcs[_]['all_pairs'] = adjs, all_pairs 
		output_arcs[_]['words'] = words 
		output_arcs[_]['vtoi'], output_arcs[_]['itov'] = vtoi, itov 
		output_arcs[_]['golds'] = golds_out

"""
with open('data/batch_'+str(batch_id)+'.pkl', 'wb') as handle:
with open('data/sent_18.pkl', 'wb') as handle:
    pickle.dump(output_arcs, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

"""
if spacy 
""" 

#nlp = spacy.load("en_core_web_sm")

#doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)
"""
import spacy

nlp = spacy.load("en_core_web_sm")
def spacy_deps(doc):
	tups = [] 
	for tki, token in enumerate(doc):
		dep = token.text +"_"+str(tki)
		head = token.head.text+"_" +str(token.head.i)
		arc = token.dep_ 
		tups.append((dep, head, arc))
	return tups 


def AddEdges(depG, deps):
	for tup in deps:
		# dependent, head, arc 
		depG.add_edge(tup[0], tup[1], label=tup[2])

deps = [] 
for sent in orig_sents:
	doc = nlp(sent)
	orig_dep = spacy_deps(doc) 
	deps.append(orig_dep)
"""

#tmpG = nx.DiGraph(directed=False) 
#AddEdges(tmpG, tmp, vtoi) 
#A = AdjGraph(tmpG) 
#e = (2,3,{'weight':7}) 

