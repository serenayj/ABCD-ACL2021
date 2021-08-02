import numpy as np
import time
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn

logger = logging.getLogger(__name__)


def reverse_padded_sequence(inputs, lengths, batch_first=False):
	"""Reverses sequences according to their lengths.
	Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
	``B x T x *`` if True. T is the length of the longest sequence (or larger),
	B is the batch size, and * is any number of dimensions (including 0).
	Arguments:
		inputs (Variable): padded batch of variable length sequences.
		lengths (list[int]): list of sequence lengths
		batch_first (bool, optional): if True, inputs should be B x T x *.
	Returns:
		A Variable with the same size as inputs, but with each sequence
		reversed according to its length.
	"""
	if batch_first:
		inputs = inputs.transpose(0, 1)
	max_length, batch_size = inputs.size(0), inputs.size(1)
	if len(lengths) != batch_size:
		raise ValueError('inputs is incompatible with lengths.')
	ind = [list(reversed(range(0, length))) + list(range(length, max_length))
		   for length in lengths]
	ind = Variable(torch.LongTensor(ind).transpose(0, 1))
	for dim in range(2, inputs.dim()):
		ind = ind.unsqueeze(dim)
	ind = ind.expand_as(inputs)
	if inputs.is_cuda:
		ind = ind.cuda(inputs.get_device())
	reversed_inputs = torch.gather(inputs, 0, ind)
	if batch_first:
		reversed_inputs = reversed_inputs.transpose(0, 1)
	return reversed_inputs

def DisSentPool(pool_type, sent_len, sent_output):
	if pool_type == "mean":
		sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1)
		emb = torch.sum(sent_output, 0).squeeze(0)
		emb = emb / sent_len.expand_as(emb)
	elif pool_type == "max":
		emb = torch.max(sent_output, 0)[0]
	return emb


class BLSTMEncoder(nn.Module):
	def __init__(self, word_emb_dim, hidden_size, device):
		super(BLSTMEncoder, self).__init__()
		self.word_emb_dim = word_emb_dim
		self.enc_lstm_dim = hidden_size
		self.tied_weights = False
		self.biflag = None 
		self.device = device 
		bidrectional = True if not self.tied_weights else False

		logger.info("tied weights = {}, using biredictional cell: {}".format(self.tied_weights, bidrectional))
		self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
								bidirectional=bidrectional, dropout=0.1)
		self.emb_drop = nn.Dropout(0.1)

	def is_cuda(self):
		# either all weights are on cpu or they are on gpu
		return 'cuda' in str(self.device)
		#return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

	def forward(self, sent_tuple):
		# sent_len: [max_len, ..., min_len] (bsize)
		# sent: Variable(seqlen x bsize x worddim)
		sent, sent_len = sent_tuple

		# Sort by length (keep idx)
		sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
		idx_unsort = np.argsort(idx_sort)

		# uncomment if the input is np array 
		#idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
		#    else torch.from_numpy(idx_sort)
		idx_sort = torch.from_numpy(idx_sort).to(self.device)
		sent = sent.index_select(1, idx_sort)

		# apply input dropout
		# sent = self.emb_drop(sent)

		# Handling padding in Recurrent Networks
		len_tmp=torch.tensor([i for i in sent_len]).to(self.device) # prevent given numpy array are negative error 
		sent_packed = nn.utils.rnn.pack_padded_sequence(sent, len_tmp)
		#print("sent packed ", sent_packed )
		sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
		sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0] # L X B X H 
		if self.tied_weights:
			# we also compute reverse
			sent_rev = reverse_padded_sequence(sent, sent_len)
			sent_rev_packed = nn.utils.rnn.pack_padded_sequence(sent_rev, sent_len)
			rev_sent_output = self.enc_lstm(sent_rev_packed)[0]
			rev_sent_output = nn.utils.rnn.pad_packed_sequence(rev_sent_output)[0]
			back_sent_output = reverse_padded_sequence(rev_sent_output, sent_len)
			sent_output = sent_output + back_sent_output

		# Un-sort by length, uncomment if the input is np array 
		#idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
		#    else torch.from_numpy(idx_unsort)
		idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
		sent_output = sent_output.index_select(1, Variable(idx_unsort)) # L X B X H 

		# Pooling used in Allen Nie's work 
		# Comment out for GCN layer 
		"""
		if self.pool_type == "mean":
			sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1).cuda()
			emb = torch.sum(sent_output, 0).squeeze(0)
			emb = emb / sent_len.expand_as(emb)
		elif self.pool_type == "max":
			emb = torch.max(sent_output, 0)[0]

		return emb
		"""
		return sent_output 