import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from gat import * 
from encoder import * 
from dep_arcs import DEPARCS 
from data import *
from utils import * 

class Trainer(object):
	""" ABCD Model Definition"""
	def __init__(self, cfg, word_dim, hidden_dim, arc_dim, num_heads, lstm_dp, weight_loss, device):
		self.enc = BLSTMEncoder(word_dim, hidden_dim, device).to(device)
		# self.enc_lstm = nn.LSTM(word_dim, hidden_dim, 1,
		# 						bidirectional=bilstm, dropout=lstm_dp) 
		self.gat = GAT(2*hidden_dim, hidden_dim, arc_dim, arc_dim, num_heads).to(device)
		if cfg["classifer"] == "Bilinear":
			self.classifer = BilinearClassifier(2*hidden_dim, arc_dim, 4).to(device) #num_classes
			print("Currently using BILINEAR CLASSIFIER")
		else:
			if "multi_layer" in cfg:
				self.classifer = Classifier(2*2*hidden_dim+arc_dim, cfg["multi_layer"]).to(device)
			else:
				self.classifer = Classifier(2*2*hidden_dim+arc_dim).to(device)
		self.cfg = cfg
		self.arcs = DEPARCS 
		self.arcids = {v:k for k,v in DEPARCS.items()} 
		self.weight_loss = weight_loss 
		self.optimizer = None 
		self.batch_loss = torch.tensor(0).float().to(device) 
		self.device = device 
		if self.weight_loss:
			self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cfg["inverse_label_weights"]).to(device)).to(device) 
		else:
			self.loss_fn = nn.CrossEntropyLoss().to(device) 

		self.pos_flag = self.cfg.get("position_flag", True) 
		if self.pos_flag:
			self.pos_enc = PositionalEncoder(word_dim, self.device) 

	def get_parameters(self):
		self.enc_params = list(self.enc.parameters())
		self.gat_params = list(self.gat.parameters())
		self.cls_params = list(self.classifer.parameters())
		self.model_params = self.enc_params + self.gat_params + self.cls_params 

	def create_optimizer(self):
		if "optimizer_type" not in self.cfg:
			# default Adam 
			self.optimizer = optim.Adam(self.model_params, 
					   lr=self.cfg['lr'], 
					   weight_decay=self.cfg['weight_decay'])
		else:
			opt_type = self.cfg["optimizer_type"]
			if opt_type == "SGD":
				self.optimizer = torch.optim.SGD(
						self.model_params, lr=self.cfg['lr'],
						momentum=self.cfg.get("momentum", 0.9),
						weight_decay=self.cfg.get("weight_decay", 0.0))
			elif opt_type == "Adadelta":
				self.optimizer = torch.optim.Adadelta(self.model_params, lr=self.cfg['lr'])
			elif opt_type == "RMSprop":
				self.optimizer = torch.optim.RMSprop(self.model_params, lr=self.cfg['lr'])
			else:
				raise NotImplementedError(
					"Not supported optimizer [{}]".format(opt_type))

	def init_onehot(self):
		# initialize onehot matrix 
		label_oh, labels = EncodeOnehot(self.arcs) 
		self.label_oh = label_oh 

	def prepare(self):
		# get list of parameters and create optimizer for it  
		print("PREPARING TRIANING OPTIMIZER")
		self.get_parameters()
		self.create_optimizer() 

	def train(self):
		if self.optimizer is None:
			self.prepare()
		self.enc.train()
		self.gat.train()
		self.classifer.train()

	def eval(self):
		self.enc.eval()
		self.gat.eval()
		self.classifer.eval()

	def save_model(self, path, prefix):
		torch.save(self.enc.state_dict(), path+"/"+prefix+ "_enc.pt")
		torch.save(self.gat.state_dict(), path+"/"+prefix+"_gat.pt")
		torch.save(self.classifer.state_dict(), path+"/"+prefix+"_clsf.pt") 


	def update(self):
		""" Update the network
		Args:
			loss: loss to train the network; dict()
		call outside the trainer, in main function 
		"""
		#self.it = self.it + 1
		self.optimizer.zero_grad() # set gradients as zero before update

		self.batch_loss.backward(retain_graph=True)
		#if self.scheduler is not None: self.scheduler.step()
		if self.cfg["gradient_clip"]:
			torch.nn.utils.clip_grad_norm_(self.model_params, 2.0)
		self.optimizer.step()
		#self.optimizer.zero_grad()
		self.batch_loss = torch.tensor(0).float().to(self.device) 

	def extract(self, sents, pairs):
		h_srcs, h_tgts, h_arcs= [], [], [] 
		for tup in pairs:
			h_srcs.append(sents[tup[0]])
			h_tgts.append(sents[tup[1]]) 
			vecs = tup[-1]
			h_arcs.append(vecs) 
		srcs = torch.stack(h_srcs, dim=0)
		tgts = torch.stack(h_tgts, dim=0)
		arcs = torch.stack(h_arcs, dim=0)
		return srcs, tgts, arcs 

	def main(self, sents, lengths, adj_pairs, golds, mode="Train"):
		if self.pos_flag:
			sents = self.pos_enc(sents.unsqueeze(0)) 
		sent_hidden = self.enc((sents.squeeze(0).unsqueeze(1), lengths)).float() # length x bsize x h 
		srcs, tgts, arcs = self.extract(sent_hidden.squeeze(1), adj_pairs)  
		h_att2, h_src, h_tgt, h_arc = self.gat(srcs, tgts, arcs)
		preds = self.classifer(h_att2, h_src, h_tgt, h_arc )
		#golds = golds.to(self.device)
		if mode == "Train":
			golds = golds.to(self.device)
			self.batch_loss += self.loss_fn(preds, golds)
		return preds

	def constructgraph(self, preds, adj_pairs, adjs, itov):
		pred_ind = torch.max(preds, dim=1)[1]
		a, b, c, d = [],[],[],[] 
		pred_pairs = [] 
		for p in adj_pairs:
			_ind = adj_pairs.index(p)
			src, tgt, arc = p[0], p[1], pred_ind[_ind].item() 
			if arc == 0:
				a.append((src, tgt, arc))
			elif arc == 1:
				b.append((src, tgt, arc))
			elif arc == 2: 
				c.append((src, tgt, arc))
			else:
				d.append((src, tgt, arc))

		output_strs = PredictGraph(a, b, c, d, adjs, itov) 
		return output_strs 


