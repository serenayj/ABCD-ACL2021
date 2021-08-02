"""
debug process pipeline 
"""

import sys 
import pickle 

from nltk import sent_tokenize 
from io import StringIO
import networkx as nx 

from data import *
from utils import * 

#import stanfordnlp
#nlp = stanfordnlp.Pipeline()

sents = open("toy_data/test.txt").readlines()
orig_sents = [s.split("<#####>")[0] for s in sents] 


lines = open("/Users/sailormoon/Downloads/stanford-corenlp-4.1.0/test.txt.out").readlines() 

loc = lines.index("Dependency Parse (enhanced plus plus dependencies):\n") 
dep_lines = lines[loc+1:]

"""
if stanford nlp 
"""

def GetDeps(dep_lines):
	tups = [] 
	words = [] 
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

		for i in [src, tgt]:
			if i not in words and i != '':
				words.append(i) 
		if src == '' or tgt == '':
			continue 
		else:
			tups.append((src, tgt, arc))
	return tups, words 

tmp, words = GetDeps(dep_lines)
vtov, vtoi = wordict(words) 
itov = {v:k for k,v in vtoi.items()}
adjs, conn_adjs = BuildAdj(vtoi, tmp) # connected nodes by neighbor, adj: nodes connected by dependency arcs
connA, AG = BuildGraph(conn_adjs, mode="Conn") # returns two things: connA (sorted by nodes), connected adjcency matrix; graph object,  

all_pairs = BuildPairs(adjs) 
#tmp = GetDeps(dep_lines)

from encoder import *
enc = BLSTMEncoder(300, 300, torch.device("cpu")) 

sent = torch.randn(15,1,300) # L X B X H 
out = enc((sent, np.array([15])))

from dep_arcs import DEPARCS
arcs = DEPARCS
label_oh, labels = EncodeOnehot(arcs) 

from trainer import Trainer
cfg = None 
bot = Trainer( cfg, 300, 300, 60, 2, 0.1, False, torch.device("cpu"))
bot.init_onehot()
h_src,h_tgt, h_arc = bot.extract(sent[:,0,:], all_pairs) # nodes x H 
out = bot.gat(h_src, h_tgt, h_arc)
new_arc = out * h_arc.squeeze(1)

from gat import *
clsf = Classifier(60)
decision = clsf(new_arc.float())

tmp = "Sokuhi was born in Fuzhou , Fujian , China. Sokuhi was ordained at 17 by Feiyin Tongrong."
from nltk import sent_tokenize
golds = sent_tokenize(tmp)
from distant_superv import *
new_temp, cut_temp, del_temp =  KEEP(all_pairs, golds, itov, tmp) 
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

