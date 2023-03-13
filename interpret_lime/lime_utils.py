from args_utils import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from torch.optim import Adam
from scipy.spatial.distance import cosine
import sklearn
import tqdm
import copy
import time
from sklearn.linear_model import Ridge

from DATA_CONFIGS import *
# generate current batch of perturbations
def generate_perturbed_sentences(seed_sample, num_perturb, max_length, tokenizer, device, backbone):
	# initialize a bow form
	vocab = list(set( [token for token in seed_sample if token not in SPECIAL_TOKENS[backbone]] ))
	#print(seed_sample)
	sentence_without_specials = [token for token in seed_sample if token not in SPECIAL_TOKENS[backbone]]
	
	mod_sentence = tokenizer.convert_ids_to_tokens(sentence_without_specials)
	bow = np.zeros((num_perturb, len( vocab ) ))
	batch_sequence, batch_attention = [],	[]
	mapper = np.zeros((len(vocab), len(sentence_without_specials)))
	for i, word in enumerate(sentence_without_specials):
		mapper[vocab.index(word), i] += 1.0
	doc_size = len(vocab)
	sample = np.random.randint(1, doc_size + 1, num_perturb - 1)
	data = np.ones((num_perturb , doc_size))
	data[0] = np.ones(doc_size)
	features_range = range(doc_size)
	inverse_data = [sentence_without_specials]
	
	for i, size in enumerate(sample, start=1):
						inactive = np.random.choice(features_range, size,
																								replace=False)
						data[i, inactive] = 0
						inverse_data.append([x for i, x in enumerate(vocab) if i not in inactive])
	
	for sampled_tokens in inverse_data:
		
		attention = [1]*(len(sampled_tokens) + 2)
		num_pads = (max_length - (len(sampled_tokens) + 2))
		sampled_tokens = [START[backbone]] + sampled_tokens + [END[backbone]] + [PAD[backbone]]*num_pads
		attention = attention + [0]*num_pads
		
		batch_sequence.append(torch.LongTensor( sampled_tokens ).unsqueeze(0).to(device))
		batch_attention.append(torch.LongTensor( attention ).unsqueeze(0).to(device))
	
	return	batch_sequence, batch_attention, data, mod_sentence, mapper


def get_distance_metric(embedding1, embedding2):
	#print(embedding2[0, ...].reshape(1, -1).shape)
	return sklearn.metrics.pairwise.pairwise_distances(
								embedding2, embedding2[0, ...].reshape(1, -1) , metric='cosine').ravel() 
	#return np.array([cosine(embedding1.cpu().detach().numpy(), embedding2[i, ...].cpu().detach().numpy()) for i in range(embedding2.size(0))])
	

def interpret_linear_model(mapper, vocab_weight, pred_label):
	#print(mapper.shape, vocab_weight.shape)
	w_vec = np.dot(mapper.T, np.squeeze(vocab_weight))
	
	#x_prob = w_vec[:, 0]
	
	return w_vec

def get_orig_model_outputs(inputs, model, mode):
	orig_output, orig_embedding = model(inputs, flag = 'test', mode = mode, interpret_mode='lime')
	orig_output = orig_output.detach()
	orig_embedding = orig_embedding.detach()
	return orig_output, orig_embedding
	
def lime_per_example(inputs, model, args, tokenizer):
		'''
		x : sentence to be interpreted 
		
		'''
		with torch.no_grad():
			orig_output, orig_embedding = get_orig_model_outputs(inputs, model, args.baseline)
		batch_sequence, batch_attention, bow, mod_sentence, mapper = generate_perturbed_sentences(inputs['input_ids'].squeeze(0).tolist(), args.num_perturb, args.max_sent_len, tokenizer, args.device, args.backbone)
		batch_sequence = torch.cat(batch_sequence, dim = 0)
		batch_attention = torch.cat(batch_attention, dim = 0)
		#local_linear_model = local_lm(bow.shape[1], args.class_num).to(args.device)
		num_batches = math.ceil(batch_sequence.size(0)/(args.per_gpu_eval_batch_size*1.0))
		
		
		model_output, model_embedding = [], []
		with torch.no_grad():
			start = 0
			end = args.per_gpu_eval_batch_size
			for bz in range(num_batches):
					if end > args.num_perturb:
						end = args.num_perturb
					
					input_ids = batch_sequence[start:end]
					attentions = batch_attention[start:end]
					input_dict = {}
					input_dict['input_ids'] = input_ids
					input_dict['attention_mask'] = attentions
					model_output_b, model_embedding_b = model(input_dict, flag = 'test', mode = False, interpret_mode = 'lime')
					model_output.append(model_output_b)
					model_embedding.append(model_embedding_b)
					start = end
					end = end + args.per_gpu_eval_batch_size
			model_output = torch.cat(model_output, dim = 0)
			model_embedding = torch.cat(model_embedding, dim = 0)
					
		#print('*'*10 , 'new sample: ')
		local_linear_model = Ridge(alpha=1, fit_intercept=True, random_state = args.seed)
		distance_metric = get_distance_metric(orig_embedding.cpu().detach().numpy(), model_embedding.cpu().detach().numpy())
		neighborhood_labels = torch.max(model_output, dim = -1)[1] .cpu()
		labels_column = np.zeros((neighborhood_labels.shape[0], int(orig_output.detach().cpu().numpy().shape[1])))
		labels_column[np.arange(neighborhood_labels.shape[0]),neighborhood_labels ] = 1
		#print(bow, labels_column[:, np.argmax(orig_output.detach().cpu().numpy(), axis=1) ], distance_metric)
		#print(orig_output.detach().cpu().numpy().shape)
		local_linear_model.fit(bow, labels_column[:, np.argmax(orig_output.detach().cpu().numpy(), axis=1) ], distance_metric)
		
		vocab_weight = local_linear_model.coef_

		'''
		for epoch in range(40):
				start = 0
				end = args.per_gpu_eval_batch_size
				total_loss = 0
				for bz in range(num_batches):
					if end > args.num_perturb:
						end = args.num_perturb
					local_linear_model.zero_grad()
					optimizer.zero_grad()
					input_ids = batch_sequence[start:end]
					attentions = batch_attention[start:end]
					input_dict = {}
					input_dict['input_ids'] = input_ids
					input_dict['attention_mask'] = attentions
					model_output, model_embedding = model(input_dict, flag = 'test', mode = False, interpret_mode = 'lime')
					linear_output = local_linear_model(torch.Tensor( bow[start:end, :] ).to(args.device))
					#print(torch.log_softmax(linear_output, dim = -1).size(), torch.max( orig_output, dim = -1)[1])
					classification_loss = F.nll_loss(torch.log_softmax(linear_output, dim = -1), torch.max(model_output, dim = -1)[1] , reduction = 'none') # nll_loss [input, target]
					
					distance_metric = get_distance_metric(orig_embedding, model_embedding)
					#print(classification_loss, distance_metric)
					total_epoch_loss = classification_loss * torch.Tensor(distance_metric).to(args.device)
					
					total_epoch_loss.sum().backward()
					total_loss += total_epoch_loss.sum()
					#nn.utils.clip_grad_norm_(local_linear_model.parameters(), max_norm=args.clip)
					optimizer.step()
					start = end
					end = end + args.per_gpu_eval_batch_size
				total_loss = total_loss/num_batches
				if total_loss < best_val_loss:
					vocab_weight = local_linear_model.linear.weight.data.cpu().numpy().T
					best_val_loss = total_loss
				print(total_loss)
		# A, vocab_weight, pred_label
		print('*'*10)
		'''
		word_score = interpret_linear_model(mapper, vocab_weight, np.argmax(orig_output.detach().cpu().numpy(), axis=1) )
		x_prob = copy.deepcopy(word_score)
		for i in range(len(word_score)):
			x_prob[i] = 1 / (1 + math.exp(-word_score[i]))
		return mod_sentence, word_score, x_prob, np.argmax(orig_output.detach().cpu().numpy(), axis=1), torch.softmax(orig_output.detach().cpu(), dim = -1)[:,np.argmax(orig_output.detach().cpu().numpy(), axis=1)].numpy()[0][0]
		
				
				
def interpret_with_lime(test_dataset, labels, model, tokenizer, args ):
	#model_name = args.save_name.replace('test_only_True_', '')
	#save_interpretation_file = open( os.path.join(args.save_dir, model_name , 'interpret.txt'), 'w' )
	
	save_interpretation_file = args.interpret_file
	interpretation_file_object = open(save_interpretation_file, 'w')
	interpretation_file_object.close()
	with open(save_interpretation_file, 'a') as f:
			
				f.write('original_sentences >> tokenized_sentences >> linear_weights >> weights_probability >> pred_probability >> true_label >> pred_label >> time\n')
			
	sample_no = 1
	
	for sample_no, (sentence, label) in (enumerate(zip( test_dataset, labels))):
		
		if len(sentence.split())>2:
			#print(sentence)
			start = time.time()
			inputs = tokenizer(sentence, padding='max_length', truncation=True, return_tensors="pt", max_length = args.max_sent_len)
			data_sample =	{
									'input_ids': inputs['input_ids'].to(args.device),
									'attention_mask': inputs['attention_mask'].to(args.device),
									'labels': torch.Tensor([label]).long().to(args.device),
									}
		
			sample_no += 1
			mod_sentence, word_score, x_prob, pred, pred_prob = lime_per_example(data_sample, model, args, tokenizer)
			end = time.time()
			with open(save_interpretation_file, 'a') as f:
			
				f.write(sentence + ' >> ' + ','.join([word for word in mod_sentence]) \
					+ ' >> ' + ','.join([str(w) for w in word_score]) \
						+' >> ' + ','.join([str(w) for w in x_prob]) \
						+ ' >> ' + str(pred_prob)	+ ' >> ' + str(label) + ' >> ' + str(pred[0]) \
								+ ' >> ' + str(end - start) + '\n')
			print('time for one sample of length ', len(sentence.split()),' =	', end -	start)
			#print(sample_no)
			if sample_no == args.num_examples:
				quit()

		



					


						



