import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import *
import numpy as np 
from nltk import word_tokenize
import random 
import logging
import pandas as pd  
import argparse 
from trainer import Trainer 
from data import *
import csv 
import json 
import pickle 
from datetime import date 
today = str(date.today()) 

class SupervisedTrainer(object):
	"""SupervisedTrainer for running ABCD model"""
	def __init__(self, config, train_db, test_db, setting_prefix=None, write_csv=False):
		super(SupervisedTrainer, self).__init__()
		self.config = config 
		self.train_db = train_db 
		self.test_db = test_db
		self.model = Trainer(config, config["word_dim"], config["hidden_dim"], len(DEPARCS), config["num_heads"], config["dropout"], config["weight_label"], torch.device(config["device"])) 
		self.batch_size = config['batch_size']
		self.epoch = config['epoch']
		self.save_loss = 1000
		self.lr_adj = config['lr_adj'] 
		self.prefix = config['dataset'] 
		if "classifer" in self.config:
			self.prefix += "_"+self.config["classifer"]
		if setting_prefix:
			self.prefix += "_" + setting_prefix   
		print("EXPERIMENT SETTING ===== ", self.prefix)
		self.init_save_dir() 
		self.best_score = 0 
		self.write_csv = write_csv
				
	def init_save_dir(self):
		today = str(date.today())
		self.dir_name = self.config["root_dir"] + self.prefix + "_"+today
		print("EXP RECORDED DATE: ", today) 
		if os.path.exists(self.dir_name):
			pass
		else:
			os.mkdir(self.dir_name) 
	
	def train(self):
		""" Run training network """
		eval_every = self.config.get("every_eval", 2) # epoch
		eval_after= self.config.get("after_eval", 1) # number of epochs starting to eval after  
		#num_step = self.config["optimize"].get("num_step", 30) # epoch
		#apply_cl_after = self.config["model"].get("curriculum_learning_at", -1)

		ii = 1
		self.model.train() # set network as train mode

		self.train_db_lst = len(self.train_db) 
		n_iters = int(self.train_db_lst / self.batch_size) 
		print("=====> # of iteration per one epoch: {}".format(n_iters))

		for epoch in range(0, self.epoch):
			start = time.time() 
			if epoch != 0 and epoch % 60 ==0 and self.lr_adj:
				self.model.update_lr()

			epoch_loss =0 
			print("Shuffling batch with {} iterations ".format(n_iters))
			permutation = np.random.permutation(self.train_db_lst) # D, length of D 
			permutation = list(map(int,permutation))
			self.permutation = permutation
			self.model.enc.train()
			self.model.gat.train()
			self.model.classifer.train()
			for _iter in range(n_iters):
				batched = [] 
				for _id in permutation[_iter*self.batch_size: (_iter+1)*self.batch_size]:
					sample = self.train_db[_id]
					# Drop sentence with length more than 80 words
					if len(sample['all_words']) >= 80:
						continue 
					sent, length, adj_pair, golds = sample['sent'], np.array([sample['sent'].shape[0]]), sample['pair_vecs'], sample["gold_labels_tensor"]
					preds = self.model.main(sent, length, adj_pair, golds)
				
				epoch_loss = self.model.batch_loss.item() 
				self.model.update()
			end = time.time() 

			print("[TRAINING] EPOCH {} BATCHWISE LOSS : {:.5f}, TOOK TIME {} ".format(epoch, epoch_loss/self.batch_size, end-start))
			# iteration done

			#Save models if best scores 
			if epoch > eval_after and epoch % eval_every == 0:
				print("VALIDATING TRAINING BATCH SEE IF OVERFIT =======")
				new_best_score = self.validate_translate(epoch, self.best_score, self.write_csv)
				if new_best_score > self.best_score:
					print("Saving Model with Best Scores: ", new_best_score)
					#self.save_loss = self.model.batch_loss.item()
					name = self.prefix+str(new_best_score)[:4]
					self.model.save_model(self.dir_name, name)
					self.best_score = new_best_score
			#self.model.train()
		self.model.save_model(self.dir_name, "best")				

	def validate_translate(self, epoch, best_score, write=False):
		start = time.time() 
		#self.model.eval()
		self.test_db_lst = len(self.test_db.data.keys())
		n_iters = int(self.test_db_lst / self.batch_size)
		batches = [] 
		permutation = list(self.test_db.data.keys())
		scores = [] 
		output = {} 
		f_scores = [] 
		for _k in permutation:
			sample = self.test_db[_k]
			output[_k] = {} 
			sent, length, adj_pair, golds = sample['sent'], np.array([sample['sent'].shape[0]]), sample['pair_vecs'], sample["gold_labels_tensor"]
			preds = self.model.main(sent, length, adj_pair, golds,mode="Valid") 
			pred_strs = self.model.constructgraph(preds, sample['adj_pairs'], sample['adj'], sample['itov'])
			gold_strs = sample["gold_sent"]
			output[_k]["pred_labels"] = preds.detach().cpu().numpy()  
			output[_k]["pred_strs"] = pred_strs
			output[_k]["gold_strs"] = gold_strs
			self.gold_labels = sample['gold_labels_tensor'].cpu().numpy() 
			self.pred_labels = torch.max(preds, dim=1)[1] 
			if self.pred_labels.shape[0] == self.gold_labels.shape[0]: 
				self.rec = get_multiclass_recall(self.pred_labels.detach().cpu().numpy(),self.gold_labels, 4) 
				self.prec = get_multiclass_prec(self.pred_labels.detach().cpu().numpy(),self.gold_labels, 4)
				f_sc = f1_avg(self.rec, self.prec) 
				f_scores.append(f_sc)
				try:
					bleu, jacc = EvalStrs(pred_strs, gold_strs)
					sc = (bleu + jacc) / 2
					scores.append(sc)
				except:
					continue  
			else:
				continue 
		end = time.time()  
		print("[VALIDATION] EPOCH {} MEAN BLEU SCORES : {:.5f}, MEAN F1 SCORES:  {:.5f},  TOOK TIME {} ".format(epoch, np.mean(scores), np.mean(f_scores), end-start))
		self.model.train() 
		new_best_score = np.mean(f_scores) 
		if new_best_score > best_score and write:
			with open(self.dir_name+"/valid_output_best.pkl", 'wb') as file:
				pickle.dump(output, file)
		return new_best_score 


if __name__ == "__main__":
	cfg = {"dataset":"MinWiki_MatchVP",
			"use_cuda": True, 
			"device": "cuda", 
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
			"classifer": "Bilinear", 
			"gradient_clip":None, 
			"root_dir":"/export/home/yug125/Complex_graph/", 
			"glove_dir": "/export/home/yug125/rl-dep-edu/", 
			"inverse_label_weights":[0.01671244, 0.35338219, 0.41641111, 0.21349426]}  
	
	start = time.time() 
	# Train Dataloader 
	train_data = ComplexSentenceDL(cfg["root_dir"]+"data/matchvp/", cfg["glove_dir"]+"glove.6B.100d.txt", cfg["use_cuda"], "Train")
	train_data.Loading()
	# Eval Dataloader  
	test_data = ComplexSentenceDL(cfg["root_dir"]+"data/", cfg["glove_dir"]+"glove.6B.100d.txt",cfg["use_cuda"], "Test") 
	test_data.Loading()
	end = time.time()
	setting_prefix = str(cfg["lr"])+"_main_ep"+str(cfg["epoch"])+"_hdim"+str(cfg["hidden_dim"])
	print("==== FINISHING LOADING DATASET, TOOK {} SECONDS =====".format(end-start))
	bot = SupervisedTrainer(cfg, train_data, test_data, setting_prefix, True) 
	bot.train() 
	
