
where_it_from = {}
nodes = list(itov.keys())
for n in nodes:
	node = itov[n]
	where_it_from[n] = [] 
	for g in golds:
		if node.split("-")[0] in word_tokenize(g.lower()):
			where_it_from[n].append(g)


used = []
for g in golds:
	for i in all_pairs:
		if i in used:
			continue 
		src, tgt = i[0], i[1] 
		srctk, tgttk = itov[src].split("-")[0], itov[tgt].split("-")[0]
		if srctk in g and tgttk in g:
			print(" Accept, ", srctk, tgttk)
			used.append(i)
		elif srctk in g and tgttk not in g and tgttk in " ".join(list(set(golds))):
			print(" Break, ", srctk, tgttk)
			used.append(i)
		elif tgttk in g and srctk not in g and srctk in " ".join(list(set(golds))):
			print(" Break, ", srctk, tgttk)
			used.append(i)

used = [] 
accept = [] 
for i in all_pairs:
	if i in used:
		continue 
	src, tgt = i[0], i[1] 
	src_in_golds, tgt_in_golds = where_it_from[src], where_it_from[tgt]
	if set(src_in_golds).intersection(set(tgt_in_golds)):
		accept.append(i)
		used.append(i) 

lefts = []
for i in all_pairs:
	if i not in accept:     
		lefts.append(i)

breaks, drops = [], []  
for i in lefts:
	src, tgt = i[0], i[1] 
	if src == tgt and where_it_from[src] == []:
		drops.append(i)
	elif where_it_from[src] == []:
		drops.append(i)
	elif where_it_from[tgt] == []:
		drops.append(i)
	else:
		breaks.append(i)

