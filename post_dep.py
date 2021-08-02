"""
Scripts to post-process dependency parses, to remove sentence with wrong identifiers, generate clean:
- orig files
- gold files
- dep files 
"""

batch_id = 0 
for batch_id in range(0,21):
	deplines = open("orig_sent_"+str(batch_id)+".txt.out").readlines() 
	locs = [ind for ind, value in enumerate(deplines) if "Sentence #" in value]
	proce_sents = [deplines[l +1] for l in locs]

	lines = open("orig_sent_"+str(batch_id)+".txt").readlines()
	sents = {k:v for k,v in enumerate(lines)} 

	golds = open("gold_sents_"+str(batch_id)+".txt").readlines()
	gold_sents = {k:v for k,v in enumerate(golds)}

	orig_prob, dep_prob = [], []  
	with open("clean_orig_sent_"+str(batch_id)+".txt", "w") as outf: 
		for k,v in sents.items():
			flag = False 
			for t in proce_sents:
				if flag == True:
					break 
				if v.strip() == t.strip():
					flag = True 
					break 
			if flag == True:
				orig_prob.append(k)  
				outf.write(v)

	with open("clean_gold_sent_"+str(batch_id)+".txt", "w") as outf:
		for _id in orig_prob: 
			outf.write(gold_sents[_id])


	clean_sent_D = {lines[i].strip():i for i in orig_prob} # getting orig sentence by ids in the clean list 
	proc_sent_D = {deplines[ind+1]: ind for ind in locs}
	new_proc_IDs = [] 
	with open("clean_orig_sent_"+str(batch_id)+".txt.out",'w') as outf:
		for ss,lineid in proc_sent_D.items():
			if ss.strip() in clean_sent_D:
				new_proc_IDs.append(lineid) 
				if locs.index(lineid) != len(locs) -1:
					end = locs[locs.index(lineid)+1]
					tmp = deplines[lineid:end-1]
				else:
					tmp = deplines[lineid:] 

				for lout in tmp:
					outf.write(lout) 



