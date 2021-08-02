import sys 
import pickle 

from nltk import sent_tokenize, word_tokenize  
from io import StringIO
import networkx as nx 
from torchtext.data.metrics import bleu_score 
import nltk 
import numpy as np 

"""
Parsing utils 
"""
def spacy_deps(doc):
	tups = [] 
	for tki, token in enumerate(doc):
		dep = token.text +"_"+str(tki)
		head = token.head.text+"_" +str(token.head.i)
		arc = token.dep_ 
		tups.append((dep, head, arc))
	return tups 

def wordict(words):
	vtov = {} # key to word 
	vtoi = {} # key to index 
	words_srt = sorted(words, key=lambda x:int(x.split("-")[-1]))
	cnt = 0 
	for w in words_srt:
		# word plus new index, because the old index was messed up with punctuations removed 
		new_key = w # this is the old key from parsing, preserve to extract dependency relations 
		vtov[new_key], vtoi[new_key] = w.split("-")[0], cnt 
		cnt +=1 
	return vtov, vtoi


"""
Graph utils 
"""

def AddSelf(itov, all_pairs):
	# Add self label to help indicate what words to be dropped 
	all_nodes = list(itov.keys())
	for n in all_nodes:
		all_pairs.append((n,n,['self']))
	return all_pairs 

def AddEdges(depG, deps, vtoi):
	for tup in deps:
		# dependent, head, arc 
		#depG.add_edge(tup[0], tup[1], label=tup[2])
		src, tgt = vtoi[tup[0]], vtoi[tup[1]]
		depG.add_edge(src, tgt, label=tup[2]) 

def BuildAdj(vtoi, deps):
	"""
	Build connectivity adj matrix;  
	"""
	itov = {v:k for k,v in vtoi.items()}
	adjs = {} 
	conn_adjs = {} 
	for k, v in itov.items():
		adjs[k], conn_adjs[k] = {}, {} 
		src = itov[k] 
		for tup in deps:
			if tup[0] == src:
				tgt_k, arc = vtoi[tup[1]], tup[-1]
				adjs[k][tgt_k], conn_adjs[k][tgt_k] = [arc], 1
		if k+1 < len(itov):
			conn_adjs[k][k+1] = 1 
			if k+1 in adjs[k]: 
				adjs[k][k+1].append('neighbor')
			else:
				adjs[k][k+1]=['neighbor']
		if k > 0:
			conn_adjs[k][k-1] = 1  
			if k-1 in adjs[k]: 
				adjs[k][k-1].append('neighbor')
			else:
				adjs[k][k-1]=['neighbor']

		conn_adjs[k][k] = 1 
	return adjs, conn_adjs  

def AdjGraph(g):
	A = nx.to_numpy_matrix(g, nodelist=sorted(g.nodes())) 
	#A = nx.adjacency_matrix(g)
	return A 

def BuildGraph(adjs, mode="Conn"):
	if mode == "Dep":
		new_graph = nx.DiGraph()
		for source, targets in adjs.items():
			if len(targets) > 0: 
				for target in targets:
					new_graph.add_edge(source, target)
		conn_A = AdjGraph(new_graph) 
		return conn_A, new_graph 
	else:
		new_graph = nx.DiGraph()
		for source, targets in adjs.items():
			for target in sorted(targets.keys()):
				new_graph.add_edge(source, target)
		conn_A = AdjGraph(new_graph) 
		return conn_A, new_graph

def GetCCs(Gs):
	""" Get a list of CCs"""
	for cc in nx.weakly_connected_components(Gs):
		yield cc 

def ExamineGraph(adj, graph):
	for i in range(len(graph.nodes())):
		for j in range(len(graph.nodes())):
			r,c = i, j
			if adj[r,c] == 1:
				print("Connected: ", list(graph.nodes)[r], list(graph.nodes)[c])

def BuildPairs(adj):
	alls = [] 
	for k,v in adj.items(): # src 
		for kk, vv in v.items(): # tgt 
			tups = (k, kk, vv) 
			alls.append(tups) 
	return alls 


def PredictGraph(pa, pb, pc, pd, adjs, itov):
	"""
	Postprocess mechanism to find CCs from predicted actions  
	"""
	G=nx.DiGraph()
	for n, v in adjs.items():
		for tgt, vv in v.items():
			G.add_edge(n,tgt) 

	for item in pb: 
		e = (item[0], item[1])
		if G.has_edge(*e): 
			#print("removing edge: ", item) 
			G.remove_edge(*e)  

	tgts = [] 
	for item in pd:
		tgts.append(item[1]) 

	from collections import Counter
	c = Counter(tgts)
	for k,v in c.items():
		if v > 1:
			G.remove_node(k) 
			
	copys = pc
	CCs = list(nx.strongly_connected_components(G)) 
	#CCs = list(nx.connected_components(G))
	outs = [] 
	for c in CCs:
		ccp = c.copy()
		for i in c:
			for p in copys:
				if i == p[0]:
					ccp.add(p[1])
		outs.append(ccp)
	#print("After adding copy ", outs)
	app_outs = [] 
	for c in outs:
		if len(c) > 1:
			out_str = ""
			for j in sorted(c):
				out_str += itov[j].split("-")[0]+ " " 
			#print("out_str: ", out_str) 
			app_outs.append(out_str)
	return app_outs 


"""
Eval utils 
"""
def get_multiclass_recall(preds, y_label,n_classes):
	# preds: (label_size), y_label; (label_size)
	label_cat = range(n_classes)
	labels_accu = {}
	for la in label_cat:
		# for each label, we get the index of the correct labels
		idx_of_cat = y_label == la
		cat_preds = preds[idx_of_cat]
		if cat_preds.size != 0:
			accu = np.mean(cat_preds == la)
			labels_accu[la] = [accu]
		else:
			labels_accu[la] = []
	return labels_accu

def get_multiclass_prec(preds, y_label,n_classes):
	label_cat = range(n_classes)
	labels_accu = {}
	for la in label_cat:
		# for each label, we get the index of predictions
		idx_of_cat = preds == la
		cat_preds = y_label[idx_of_cat]  # ground truth
		if cat_preds.size != 0:
			accu = np.mean(cat_preds == la)
			labels_accu[la] = [accu]
		else:
			labels_accu[la] = []
	return labels_accu


def f1_avg(rec, prec):
	f1s = []
	for k,v in rec.items():
		if v:
			rec_val = v[0]
		else: 
			rec_val = 0 
		if prec[k]:
			prec_val= prec[k][0]
		else:
			prec_val =0 
		
		if prec_val == 0 and rec_val == 0:
			val = 0 
		else:
			val = 2*(rec_val*prec_val) / (rec_val+prec_val)
		f1s.append(val)
	return np.mean(f1s) 
	
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def loop_jaccard(golds, edits):
	sim = 0 
	for i in range(0,len(golds)):
		sim += jaccard_similarity(word_tokenize(golds[i].lower()), word_tokenize(edits[i].lower()))
	sim_sent = float(sim) / len(golds)
	return sim_sent 

def EvalStrs(pred_strs, golds):
	if golds == None:
		return 0,0 
	candidate = [word_tokenize(i.lower()) for i in pred_strs]
	references = [word_tokenize(i.lower()) for i in golds] 
	if len(candidate) == len(references):
		bleu_sc = bleu_score(candidate, references)

	else:
		bleu_sc = 0 
		for c in candidate:
			cand_sc = [] 
			for r in references:
				#print("refere", r, "candidate ", c)
				sc = nltk.translate.bleu_score.sentence_bleu(c, r)
				cand_sc.append(sc)
			bleu_sc += np.max(cand_sc) 
			#print("candidate ", [c])
			#print("references", references)
			#bleu_sc += bleu_score([c], references) 
		bleu_sc /= len(candidate)
	jaccard = loop_jaccard(golds, pred_strs) 
	return bleu_sc, jaccard  

	
