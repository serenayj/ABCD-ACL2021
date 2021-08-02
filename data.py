
from __future__ import print_function, division
import os
import glob 
import torch
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import time 
import warnings
from dep_arcs import DEPARCS
import networkx as nx 
import pickle5 as pickle

warnings.filterwarnings("ignore")

def GetGlove(vocab, glove_dict):
	#vocab_vals = list(vocab.values())
	word_vec = [] 
	for v in vocab:
		if v in glove_dict:
			word_vec.append(glove_dict[v])
		else:
			word_vec.append(glove_dict['<unk>'])
	return word_vec

def LoadGlove(glove_path):
	word_vec = {}
	with open(glove_path) as f:
		for line in f:
			word, vec = line.split(' ', 1)
			#if word in vocab:
			word_vec[word] = np.fromstring(vec, sep=' ')
	#print('Found {0}(/{1}) words with glove vectors'.format(
	#	len(word_vec), len(vocab)))
	word_vec['<unk>'] = np.mean(list(word_vec.values()),axis=0)
	return word_vec



def EncodeOnehot(labels, classes=None):
	"""
	Encoding dep arcs as one hot, src: https://github.com/thudzj/gcn_comm/blob/master/utils.py 
	"""
	if classes is None:
		classes = set(labels)
	else:
		classes = range(classes)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
					enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)),
							 dtype=np.int32)
	return labels_onehot, classes_dict 

def GetArc(all_pairs, one_arcs, label_arcs, use_cuda):
	newout = [] 
	for p in all_pairs:
		vecs = torch.zeros(1, len(label_arcs))
		src, tgt = p[0], p[1]
		for arc in p[2]: 
			arc_onehot = torch.tensor(one_arcs[label_arcs[arc.split(":")[0]]]).unsqueeze(0) 
			vecs += arc_onehot 
		if use_cuda:
			newout.append((src,tgt, vecs.cuda()))
		else:
			newout.append((src,tgt, vecs))
	return newout 

def BuildGraph(adjs):
	G=nx.DiGraph()
	for n, v in adjs.items():
		for tgt, vv in v.items():
			G.add_edge(n,tgt) 
	return G

def GoldLabel(all_pairs, a, b, c, d):
	gold_pairs = [] 
	for p in all_pairs:
		if p in c:
			tup = (p[0], p[1], 2)
		elif p in b:
			tup = (p[0], p[1], 1)
		elif p in d:
			tup = (p[0], p[1], 3) 
		else:
			tup = (p[0], p[1], 0)
		gold_pairs.append(tup)
	labels = torch.tensor([i[2] for i in gold_pairs]) 
	return gold_pairs, labels  


class ComplexSentenceDL(Dataset):
	"""Loading Complex Sentence dataset."""
	def __init__(self, root_dir, glove_path, use_cuda=False, mode="Train", transform=None, use_bert=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.mode = mode 
		self.glove_path = glove_path 
		self.use_cuda = use_cuda
		self.use_bert = use_bert 

	def __len__(self):
		return len(self.data)

	def Loading(self):
		print(self.root_dir)
		if not os.path.exists(self.root_dir):
			print("ERROR!!!! ROOT DIR NOT EXIST {}".format(self.root_dir)) 
			raise ValueError   

		self.data = {}
		print("====== INITIALIZING DATASET FROM {} PICKLE FILES =========".format(self.mode))
		start =time.time()
		cnt = 0 
		if self.mode == "Train":
			allfiles = list(glob.iglob(self.root_dir+"clean_batch*.pkl")) 			
			for file in allfiles:
				batch_num = file[file.split(".")[0].find("ch")+2:file.find(".pkl")] 
				batch_data = pickle.load(open(file, "rb"))
				for k,v in batch_data.items():
					v["index"] = batch_num[1:]+"_"+str(k)
					self.data[cnt] = v
					cnt +=1 
		else:
			file = self.root_dir+"test.pkl" 
			#try:
			#	batch_data = pickle.load(open(file, "rb")) 
			#except:
			#	import pickle5 as pickle
			batch_data = pickle.load(open(file, "rb"))
			self.data = batch_data 
		if self.use_bert is None:
			self.glove_dict = LoadGlove(self.glove_path)
		else:
			pass 
		arcs = DEPARCS
		self.arc_ones, self.arc_one_dics = EncodeOnehot(arcs) 
		self.label_arcs = {v:k for k,v in arcs.items()}
		end = time.time()
		print("FINISH LOADING DATA TOTAL TIME {:.4f} SECONDS".format(end-start)) 


	def __getitem__(self, idx):
		sample = {} 
		item = self.data[idx] 
		ws = item["words"]
		sent_tks = sorted(ws, key=lambda x:int(x.split("-")[1]))
		input_sent = [i.split("-")[0].lower() for i in sent_tks]
		if self.use_bert is None:
			sent_vecs = torch.tensor(GetGlove(input_sent, self.glove_dict))
		else:
			sent_vecs = " ".join(input_sent) 
		all_pairs = item['all_pairs']
		adjs = item['adjs']
		graph = BuildGraph(adjs)
		a, b, c, d = item['accept'], item['break'], item['copy'], item['drop'] 
		gold = item['golds']
		gold_pairs, gold_labels = GoldLabel(all_pairs, a, b, c, d)  
		pair_acs = GetArc(all_pairs, self.arc_one_dics, self.label_arcs, self.use_cuda)

		if self.use_cuda and self.use_bert is None:
			sent_vecs = sent_vecs.cuda() # L X D 
			sample['sent'], sample['itov'], sample['gold_sent'] =  sent_vecs.float(), item['itov'], gold   
		else:
			sample['sent'], sample['itov'], sample['gold_sent'] =  sent_vecs, item['itov'], gold  

		sample['a'], sample['b'], sample['c'], sample['d'] = a, b, c, d 
		sample['all_words'] = ws	
		sample['pair_vecs'], sample['adj_pairs'] = pair_acs, all_pairs
		sample['gold_pairs'] = gold_pairs 
		sample['adj']= adjs # keeping the graph structure as dictionary  
		if self.use_cuda:
			gold_labels = gold_labels.cuda()
		sample["gold_labels_tensor"] = gold_labels 

		return sample

class ComplexSentenceDL_Inference(Dataset):
	"""Loading Complex Sentence dataset."""
	def __init__(self, root_dir, filename, glove_path,  use_cuda=False, mode="Valid", transform=None, use_bert=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.filename = filename 
		self.glove_path = glove_path 
		self.use_cuda = use_cuda
		self.use_bert = use_bert 
		self.mode = mode 

	def __len__(self):
		return len(self.data)

	def Loading(self):
		print(self.root_dir)
		if not os.path.exists(self.root_dir):
			print("ERROR!!!! ROOT DIR NOT EXIST {}".format(self.root_dir)) 
			raise ValueError   

		self.data = {}
		#print("====== INITIALIZING DATASET FROM {} PICKLE FILES =========".format(self.mode))
		start =time.time()
		cnt = 0 
		file = self.root_dir+self.filename 

		batch_data = pickle.load(open(file, "rb"))
		self.data = batch_data 
		if self.use_bert is None:
			self.glove_dict = LoadGlove(self.glove_path)
		else:
			pass 
		arcs = DEPARCS
		self.arc_ones, self.arc_one_dics = EncodeOnehot(arcs) 
		self.label_arcs = {v:k for k,v in arcs.items()}
		end = time.time()
		print("FINISH LOADING DATA TOTAL TIME {:.4f} SECONDS".format(end-start)) 


	def __getitem__(self, idx):
		sample = {} 
		item = self.data[idx] 
		ws = item["words"]
		sent_tks = sorted(ws, key=lambda x:int(x.split("-")[1]))
		input_sent = [i.split("-")[0].lower() for i in sent_tks]
		if self.use_bert is None:
			sent_vecs = torch.tensor(GetGlove(input_sent, self.glove_dict))
		else:
			sent_vecs = " ".join(input_sent) 
		all_pairs = item['all_pairs']
		adjs = item['adjs']
		graph = BuildGraph(adjs)
		pair_acs = GetArc(all_pairs, self.arc_one_dics, self.label_arcs, self.use_cuda)

		if self.use_cuda and self.use_bert is None:
			sent_vecs = sent_vecs.cuda() # L X D 
			sample['sent'], sample['itov']=  sent_vecs.float(), item['itov']   
		else:
			sample['sent'], sample['itov'] =  sent_vecs, item['itov']  

		sample['all_words'] = ws	
		sample['pair_vecs'], sample['adj_pairs'] = pair_acs, all_pairs
		sample['adj']= adjs # keeping the graph structure as dictionary  
		if self.mode == "Post-Edit":
			sample['gold_strs'] = item["golds"]

		return sample


# root_dir = "data/"

# dataset = ComplexSentenceDL(root_dir, "/Users/sailormoon/Desktop/glove.6B.100d.txt", False, "Train")
# dataset.Loading()
# sample = dataset['0_9905']

# trainset = DataLoader(dataset=dataset,
#                       batch_size=4,
#                       shuffle=True,
#                       collate_fn=None, # use custom collate function here
#                       pin_memory=True)

