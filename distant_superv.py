"""
Distant supervision: align gold rewritten with input sen
""" 

from nltk import sent_tokenize, word_tokenize  
tmp = "Sokuhi was born in Fuzhou , Fujian , China. Sokuhi was ordained at 17 by Feiyin Tongrong."
import copy 

def KEEP(pairs, golds, itov, s):
	new_temp = [] 
	del_temp = [] 
	cut_temp = [] 
	drop_temp = [] 
	running_stack = copy.deepcopy(pairs)
	for i in pairs:
		src, tgt = i[0], i[1]
		srctk, tgttk = itov[src].split("-")[0], itov[tgt].split("-")[0]
		used = False 
		for g in golds:
			if srctk in g and tgttk in g:
				new_temp.append(i)
				used = True 
				break 
			elif srctk not in g and srctk in " ".join(list(set(golds) - set([g]))) and tgttk in g:
				cut_temp.append(i)
				used = True 
				break 
			elif tgttk not in g and tgttk in " ".join(list(set(golds) - set([g]))) and srctk in g:
				cut_temp.append(i)
				used = True 
				break 

		if used == False:
			del_temp.append(i)
			used = True  
			#for ni in [srctk, tgttk]:
			#if srctk not in " ".join(list(set(golds)))

			#break
	return new_temp, cut_temp, del_temp # all pairs kept 

def KEEPORDROP(all_pairs, golds, itov, s):
	where_it_from = {}
	nodes = list(itov.keys())
	for n in nodes:
		node = itov[n]
		where_it_from[n] = [] 
		for g in golds:
			if node.split("-")[0].lower() in word_tokenize(g.lower()):
				where_it_from[n].append(g)

	used = [] 
	accept = [] 
	for i in all_pairs:
		if i in used:
			continue 
		src, tgt = i[0], i[1] 
		src_in_golds, tgt_in_golds = where_it_from[src], where_it_from[tgt]
		if set(src_in_golds).intersection(set(tgt_in_golds)): # impossible to be empty set 
			accept.append(i)
			used.append(i) 

	lefts = []
	for i in all_pairs:
		if i not in accept:     
			lefts.append(i)
			
	breaks, drops = [], []  
	for i in lefts:
		src, tgt = i[0], i[1] 
		if src == tgt and where_it_from[src] == []: # self arc, and the node did not appear anywhere
			drops.append(i)
		elif where_it_from[src] == []: # src node not showing anywhere 
			drops.append(i)
		elif where_it_from[tgt] == []:
			drops.append(i)
		else:
			breaks.append(i)

	return where_it_from, accept, breaks, drops 


def COPY(all_pairs, golds, itov, where_it_from):
	#subj_rels = [] 
	SUBJARCS = ['nsubj','csubj','aux','expl','ccomp','acl','advcl']
	copy_pairs, subj_pairs = [],[] 
	for i in all_pairs:
		for arc in SUBJARCS:
			if arc in  " ".join(i[2]):
				subj_pairs.append(i) 
	#subj_pairs = [i for i in all_pairs if "nsubj" in " ".join(i[2])]   
	for i in subj_pairs:
		src, tgt = i[0], i[1]
		src_sent, tgt_sent = where_it_from[src], where_it_from[tgt]
		if set(src_sent).intersection(set(tgt_sent)) == set(tgt_sent) and set(tgt_sent) != set([]):
			copy_pairs.append(i)
		elif set(src_sent).intersection(set(tgt_sent)) == set(src_sent) and set(src_sent) != set([]):
			copy_pairs.append(i) 

	"""
	print("subject pairs", subj_pairs)
	for i in subj_pairs:
		src, tgt = i[0], i[1]
		srctk, tgttk = itov[src].split("-")[0], itov[tgt].split("-")[0]
		flag = False 
		for g in golds: # by order 
			if tgttk in g and srctk in g:
				if flag == False:
					flag = True 
					break 
			#elif srctk in copys or tgttk in copys and flag != True:
			elif tgttk in g and srctk not in g and flag == False:
				#print("left "," ".join(list(set(golds) - set([g]))) )
				#print("src tk, ", srctk, "tgttk", tgttk, "case 2 ", i )
				flag = True 
				copy_pairs.append(i)
	"""
	return copy_pairs

#def COPY(pairs, golds, itov, s) 
"""
import string 
from nltk import word_tokenize
from collections import Counter 
s = "Sokuhi was born in Fuzhou , Fujian , China , and was ordained at 17 by Feiyin Tongrong ."
c = Counter(word_tokenize(s))
gc = Counter(word_tokenize(tmp)) # tmp is a string of all gold proposisions
copys = [] 
for k,v in gc.items():
	if k not in string.punctuation and k in c:
		if v > c[k]:
			copys.append(k)  # identify words occuring in rewritten more than input sentence
"""
# copy_pairs = [] 
# for i in all_pairs:
# 	src, tgt = i[0], i[1]
# 	srctk, tgttk = itov[src].split("-")[0], itov[tgt].split("-")[0]
# 	flag = False 
# 	if srctk in copys or tgttk in copys and flag != True:
# 		for g in golds:
# 			print("current gold index ", golds.index(g))
# 			if tgttk in g and srctk not in g:
# 				print("left "," ".join(list(set(golds) - set([g]))) )
# 				print("src tk, ", srctk, "tgttk", tgttk, "case 2 ", i )
# 				flag = True 



				#if srctk in g and tgttk not in g and tgttk in " ".join(list(set(golds) - set([g]))):
				# 	flag = True 
				# 	print("src tk, ", srctk, "tgttk", tgttk, "case 1 ", i )
				# 	copy_pairs.append(i) 
				# 	break 
				# elif tgttk in g and srctk not in g and srctk in " ".join(list(set(golds) - set([g]))):
				# 	print("g ", g)
				# 	print("left "," ".join(list(set(golds) - set([g]))) )
				# 	flag = True 
				# 	print("src tk, ", srctk, "tgttk", tgttk, "case 2 ", i )
				# 	copy_pairs.append(i) 
				# 	break 
				# if flag:
				# 	break 



