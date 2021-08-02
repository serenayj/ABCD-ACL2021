
from dgl.nn.pytorch import GATConv
"""
Non-message passing framework for GAT 
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
from torch.autograd import Variable

class PositionalEncoder(nn.Module):
	def __init__(self, d_model,device, max_seq_len = 80):
		super().__init__()
		self.d_model = d_model
		
		# create constant 'pe' matrix with values dependant on 
		# pos and i
		pe = torch.zeros(max_seq_len, d_model)
		for pos in range(max_seq_len):
			for i in range(0, d_model, 2):
				pe[pos, i] = \
				math.sin(pos / (10000 ** ((2 * i)/d_model)))
				pe[pos, i + 1] = \
				math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
				
		pe = pe.unsqueeze(0)
		#print("self pe shape ", pe.shape)
		self.register_buffer('pe', pe)
		self.device = device
 
	def forward(self, x):
		# make embeddings relatively larger
		x = x * math.sqrt(self.d_model)
		#add constant to embedding
		seq_len = x.size(1)
		x = x + Variable(self.pe[:,:seq_len], \
		requires_grad=False).to(self.device)
		return x

class MultiHeadGATLayer(nn.Module):
	def __init__(self, in_dim, out_dim, arc_dim, num_heads, merge='cat'):
		super(MultiHeadGATLayer, self).__init__()
		self.heads = nn.ModuleList()
		for i in range(num_heads):
			self.heads.append(GATLayer(in_dim, arc_dim, out_dim))
		self.merge = merge

	def forward(self, src, tgt, arc):
		head_outs = [attn_head(src, tgt, arc) for attn_head in self.heads]
		#print(head_outs)
		if self.merge == 'cat':
			# concat on the output feature dimension (dim=1)
			#return torch.cat(head_outs, dim=1)
			return torch.mean(torch.cat(head_outs, dim=1), dim=1).unsqueeze(-1)
		else:
			# merge using average
			tmp = torch.stack(head_outs) 
			return torch.mean(tmp,dim=1).unsqueeze(-1)
			#return torch.mean(torch.stack(head_outs))


class GATLayer(nn.Module):
	def __init__(self, h_in_dim, arc_in_dim, out_dim):
		super(GATLayer, self).__init__()
		# equation (1)
		self.fc_src = nn.Linear(h_in_dim, out_dim, bias=False)
		self.fc_tgt = nn.Linear(h_in_dim, out_dim, bias=False)
		self.fc_arc = nn.Linear(arc_in_dim, out_dim, bias=False)
		# equation (2)
		self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('relu')
		nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
		nn.init.xavier_normal_(self.fc_tgt.weight, gain=gain)
		nn.init.xavier_normal_(self.fc_arc.weight, gain=gain)
		nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

	def edge_attention(self, src, tgt, arc):
		# edge UDF for equation (2)
		# z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1) # concatenation of two: src and tgt; could be extended to src, tgt, arcs 
		# a = self.attn_fc(z2)
		# return {'e': F.leaky_relu(a)}
 
		deps = torch.cat((tgt, arc), dim=1)  # assuiming input is tgt: H X B, arc: H_A X B 
		src_arcs = torch.cat((src, deps), dim=1)
		#print("after concatenation shape ", src_arcs.shape)
		a = self.attn_fc(src_arcs) 
		#h_arc_update = a * arc
		#return F.leaky_relu(h_arc_update) 
		return F.leaky_relu(a)

	def forward(self, h_src, h_tgt, h_arc):
		# equation (1)
		# self.g.ndata['z'] = z
		# # equation (2)
		# self.g.apply_edges(self.edge_attention)
		# # equation (3) & (4)
		# self.g.update_all(self.message_func, self.reduce_func)
		# return self.g.ndata.pop('h')
		#print("=== hsrc shape ==== ", h_src.shape)
		src = self.fc_src(h_src)
		#print("src shape ", src.shape )
		tgt = self.fc_tgt(h_tgt)
		#print("tgt shape ", tgt.shape)
		h_arc = h_arc.float() 
		#print("h arc shape", h_arc.shape)
		arc = self.fc_arc(h_arc)
		if len(arc.shape) == 3:
			arc = arc.squeeze(1)
		h_att_arc = self.edge_attention(src, tgt, arc) 
		return h_att_arc 

	# def message_func(self, edges):
	# 	# Not using: message UDF for equation (3) & (4)
	# 	return {'z': edges.src['z'], 'e': edges.data['e']}

	# def reduce_func(self, nodes):
	# 	# Not using 
	# 	# reduce UDF for equation (3) & (4)
	# 	# equation (3)
	# 	alpha = F.softmax(nodes.mailbox['e'], dim=1)
	# 	# equation (4)
	# 	h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
	# 	return {'h': h}


class GAT(nn.Module):
	def __init__(self, in_dim, hidden_dim, arc_dim, out_dim, num_heads):
		super(GAT, self).__init__()
		self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, arc_dim, num_heads)
		# Be aware that the input dimension is hidden_dim*num_heads since
		# multiple head outputs are concatenated together. Also, only
		# one attention head in the output layer.
		self.layer2 = MultiHeadGATLayer(in_dim, out_dim, arc_dim,  1)

	def forward(self, h_src, h_tgt, h_arc):
		if len(h_arc.shape) == 3:
			h_arc = h_arc.squeeze(1) 
		if len(h_src.shape) == 3:
			h_src = h_src.squeeze(1) 
		if len(h_tgt.shape) == 3:
			h_tgt = h_tgt.squeeze(1) 
		atten = self.layer1(h_src, h_tgt, h_arc)
		#print("layer 1 h shape ", h.shape) # nodes number x H, 2718 x 16 for cora  
		h_arc = atten * h_arc 
		h_arc = F.elu(h_arc)
		#print("===== before layer 2 ==== h arc", h_arc.shape)
		h_att2 = self.layer2(h_src, h_tgt, h_arc)
		return h_att2, h_src, h_tgt, h_arc  


class Classifier(nn.Module):
	def __init__(self, in_dim, multi_layer=False):
		super(Classifier, self).__init__()
		self.multi_layer = multi_layer 
		if multi_layer:
			self.layer1 = nn.Linear(in_dim, 128)
			self.layer2 = nn.Linear(128, 128)
			self.layer3 = nn.Linear(128,4)
		else:
			self.decision = nn.Linear(in_dim, 4)

	def forward(self,h_att2, h_src, h_tgt, h_arc ):
		""" infeats should be attention weights * h_arcs """
		deps = torch.cat((h_tgt, h_arc), dim=1)  # assuiming input is tgt: H X B, arc: H_A X B 
		src_arcs = torch.cat((h_src, deps), dim=1)
		infeats = h_att2 * src_arcs
		if self.multi_layer:
			out1 = self.layer1(infeats)
			out2 = self.layer2(out1)
			out = self.layer3(out2)
		else:
			out = self.decision(infeats)
		out = F.softmax(out)  
		return out 


class BilinearClassifier(nn.Module):
	r"""
	Biaffine Dependency Parser 的子模块, 用于构建预测边类别的图
	"""
	
	def __init__(self, hdim, arc_dim, num_label, bias=True):
		r"""
		
		:param in1_features: 输入的特征1维度
		:param in2_features: 输入的特征2维度
		:param num_label: 边类别的个数
		:param bias: 是否使用bias. Default: ``True``
		"""
		super(BilinearClassifier, self).__init__()
		self.bilinear = nn.Bilinear(hdim, hdim, num_label, bias=bias)
		self.lin = nn.Linear(2*hdim + arc_dim, num_label, bias=False)
	
	def forward(self, h_att2, h_src, h_tgt, h_arc):
		r"""
		:param x1: [batch, seq_len, hidden] 输入特征1, 即label-head
		:param x2: [batch, seq_len, hidden] 输入特征2, 即label-dep
		:return output: [batch, seq_len, num_cls] 每个元素对应类别的概率图
		"""
		bi_output = self.bilinear(h_src, h_tgt)
		deps = torch.cat((h_tgt, h_arc), dim=1) 
		src_arcs = torch.cat((h_src, deps), dim=1)
		infeats = h_att2 * src_arcs
		output = bi_output + self.lin(infeats)
		out = F.softmax(output) 
		return output
		
