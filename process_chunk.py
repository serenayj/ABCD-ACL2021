lines = open("MinWikiSplit_v1_INLG2019.txt").readlines()

# tgt = 0 
# cnt = 0 
# for l in range(0, 200000, 10000):
# 	line = lines[l:l+10000]
# 	sents = [i.split('<#####>')[1] for i in line]
# 	with open("gold_sent_"+str(cnt)+".txt", "w") as outf:
# 		for ll in sents:
# 			outf.write(ll+"\n")
# 	cnt +=1 

tgt = 0 
cnt = 0 
for l in range(200000, len(lines)):
	line = lines
	sents = [i.split('<#####>')[0] for i in line]
	with open("orig_test.txt", "w") as outf:
		for ll in sents:
			outf.write(ll+"\n")
	cnt +=1 


tgt = 0 
cnt = 0 
for l in range(200000, len(lines)):
	line = lines
	sents = [i.split('<#####>')[1] for i in line]
	with open("gold_test.txt", "w") as outf:
		for ll in sents:
			outf.write(ll+"\n")
	cnt +=1 
