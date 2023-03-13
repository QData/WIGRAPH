

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import math
from repr_utils import *
set_seed(1)

SMALL = 1e-08

class IMASK(nn.Module):
	def __init__(self, args):
		super(IMASK, self).__init__()

		self.device = args.device
		self.args  = args
		self.mask_hidden_dim = args.mask_hidden_dim
		self.W = nn.Parameter(torch.FloatTensor(args.imask_layers, args.embed_dim, args.embed_dim))

		self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'gelu': F.gelu}
		self.activation = self.activations['tanh']
		self.gcn_activation = self.activations[self.args.non_linearity]
		self.interaction_embeddings = args.interaction_embeddings

		self.embed_dim = args.embed_dim
		self.tau = args.tau
		self.imask_layers = args.imask_layers
		
		self.linear_layer = nn.Linear(self.embed_dim, self.mask_hidden_dim)
		self.hidden2p = nn.Linear(self.mask_hidden_dim, 2)
		self.weighted = args.weighted
		
		self.imask_dropout = args.imask_dropout
		self.bias = nn.Parameter(torch.FloatTensor( args.imask_layers, args.embed_dim))
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.W.size(1))
		self.W.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)



	def forward_sent_batch(self, embeds):
		temps = self.activation(self.linear_layer(embeds))
		p = self.hidden2p(temps)	# bsz, seqlen, dim
		return p

    
	def forward(self, x, local_mask, flag,  baseline = False):
		
		'''
		x  : word level embeddings 
		local_mask : a local word to word mask 
		flag : train / test : controls hard vs soft samples
		mask_token_embedding : the embedding for tokens that need to be masked
		'''
		
		if baseline:
			p = self.forward_sent_batch(x)
			if flag == 'train':
				r = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1]
				x_prime = r.unsqueeze(2).repeat(1,1,x.size(2)) * (x)
				
				
			
			else:
				r = F.softmax(p,dim=2)[:,:,1]
				
				x_prime = r.unsqueeze(2).repeat(1,1,x.size(2)) * (x)
				
			return x_prime, p
		#assert mask_token_embedding.shape == (1,1,768)
		assert len(x.shape) == 3
		#print(local_mask)
		assert local_mask.shape == (x.size(0), x.size(1), x.size(1), 2)
		# gumbel softmax takes in logits 
		local_mask_logits = torch.log(local_mask)
		#print(local_mask_logits.size(), local_mask_logits[0][0,1])
		# if training the model, use the 1/0 hard  
		if flag == 'train':
			local_mask = F.gumbel_softmax((local_mask_logits), hard=True,dim=3, tau = self.tau)[:,:,:,1]
		else:
			local_mask = F.softmax(local_mask_logits, dim=3)[:,:,:,1]
		
		# ensures self interaction = 1.0 
		
		local_mask = local_mask * (torch.ones_like(local_mask)-torch.eye(local_mask.size(1)).unsqueeze(0).repeat(x.size(0),1,1).to(local_mask.device))
		
		local_mask = local_mask + torch.eye(local_mask.size(1)).unsqueeze(0).repeat(x.size(0),1,1).to(local_mask.device)
		


		# making sure all are 1 at diagonal
		#assert torch.all(torch.eq(torch.diag(local_mask ), torch.ones_like(torch.diag(local_mask ) )))


		# normalize based on degree 
		degree_matrix = torch.diag_embed(local_mask.sum(dim=-1).pow(-0.5))
		local_mask = torch.matmul(degree_matrix, local_mask).matmul(degree_matrix)
		
		# AX 
		
		
		interaction_embedding = ((torch.matmul(local_mask, x))) 
		
		# AXW
		if self.weighted: 
			interaction_embedding = torch.matmul(interaction_embedding, self.W[0,...])
			interaction_embedding += self.bias[0,...]

		# sigma(AXW)
		interaction_embedding = self.gcn_activation( interaction_embedding )
		interaction_embedding = F.dropout(interaction_embedding, p = self.imask_dropout, training = self.training)
		
		for layer in range(1,self.imask_layers):
			# AX
			interaction_embedding = ((torch.matmul(local_mask, interaction_embedding)))	
			# AXW
			if self.weighted: 
				interaction_embedding = torch.matmul(interaction_embedding, self.W[layer,...])
				interaction_embedding += self.bias[layer,...]

			# sigma(AXW)
			interaction_embedding = self.gcn_activation( interaction_embedding)
			interaction_embedding = F.dropout(interaction_embedding, p = self.imask_dropout, training = self.training)
	
		
		if not self.interaction_embeddings:
			 
			
				p = self.forward_sent_batch(x)
		else:
				p = self.forward_sent_batch(interaction_embedding)

		
		if flag == 'train':
			
			#zero_indexes = (r==0).nonzero( as_tuple = True)
			r = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1]
			if not self.args.onlyA:
				
				x_prime = r.unsqueeze(2).repeat(1,1,x.size(2)) * (x + interaction_embedding)
			else:
				#print('ONLY A!!')
				
				x_prime = (x + interaction_embedding)
				
				

			#for i1,i2 in zip(zero_indexes[0].tolist(), zero_indexes[1].tolist()):
			#	x_prime[i1,i2,...] = mask_embedding
			
		else:
			
			#print(r)
			r = F.softmax(p,dim=2)[:,:,1]
			if not self.args.onlyA:
				
				x_prime = r.unsqueeze(2).repeat(1,1,x.size(2)) * (x + interaction_embedding)
			else:
				
				x_prime = (x + interaction_embedding)
				
		
		
		return x_prime, p

