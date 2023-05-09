"""

"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import torch
from utils import *
import numpy as np 
from nltk import word_tokenize
import random 
import logging
from trainer import Trainer 
from data import *
import pickle 
import tqdm 
from datetime import date 
today = str(date.today()) 

cfg = {"dataset":"MinWiki_MatchVP",
			"use_cuda": False, 
			"device": "cpu", 
			"batch_size":64, 
			"epoch":50, 
			"every_eval": 4, 
			"after_eval": 4, 
			"lr_adj":False,
			"lr":1e-4, 
			"weight_decay":0.99,
			"num_heads":4, 
			"word_dim":100, 
			"hidden_dim":800,
			"dropout":0.2, 
			"weight_label": True,
			"classifer": "MLP", # Use Bilinear or MLP
			"gradient_clip":None,
			"root_dir": "/Users/sailormoon/Downloads/ABCD_parser-main/",
			"glove_dir": "/Users/sailormoon/Desktop/", 
			"pretrained_path": "best_minwiki/", # if using pre-trained model  
			"data_filename": "batch_test.pkl",  # input filename 
			"output_str_to_file":True, # flag for if outputting clean output to txt file
			"inverse_label_weights":[0.01671244, 0.35338219, 0.41641111, 0.21349426] # inverse label weights for MinWiki since we are using MinWiki pre-trained model
			}  
	
enc_path = cfg["pretrained_path"] +"best_enc.pt"
gcn_path = cfg["pretrained_path"] +"best_gat.pt"
clsf_path = cfg["pretrained_path"] +"best_clsf.pt"

start = time.time() 
test_db = ComplexSentenceDL_Inference(cfg["root_dir"]+"data/", cfg["data_filename"], cfg["glove_dir"]+"glove.6B.100d.txt",cfg["use_cuda"], "Test")
test_db.Loading() 
model = Trainer(cfg, cfg["word_dim"], cfg["hidden_dim"], len(DEPARCS), cfg["num_heads"], cfg["dropout"], cfg["weight_label"], torch.device(cfg["device"]))

model.enc.load_state_dict(torch.load(enc_path, map_location="cpu"))
model.gat.load_state_dict(torch.load(gcn_path, map_location="cpu"))
model.classifer.load_state_dict(torch.load(clsf_path, map_location="cpu"))

start = time.time() 
batches = [] 
sent_keys = list(test_db.data.keys())
scores = [] 
output = {} 
f_scores = [] 
for _k in sent_keys:
	sample = test_db[_k]
	output[_k] = {} 
	golds = None 
	sent, length, adj_pair = sample['sent'], np.array([sample['sent'].shape[0]]), sample['pair_vecs']
	preds = model.main(sent.float(), length, adj_pair, golds,mode="Test") 
	pred_strs = model.constructgraph(preds, sample['adj_pairs'], sample['adj'], sample['itov'])
	output[_k]["pred_labels"] = preds.detach().cpu().numpy()  
	output[_k]["pred_strs"] = pred_strs

with open('output.pkl', 'wb') as handle:
	pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("====== WRITE MODEL OUTPUT TO output.pkl ======") 

if cfg["output_str_to_file"]:
	with open("output.txt", "w") as f:
		for k,v in output.items():
			out_str = " ".join([i+"." for i in v["pred_strs"]]) 
			f.write(out_str+"\n")
	print("====== WRITE OUTPUT STRINGS TO output.txt ======")


