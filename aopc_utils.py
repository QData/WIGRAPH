import numpy as np
import torch
from DATA_CONFIGS import *
def compute_perturbations(sentences, weights, pred_labels, pred_probs, tokenizer, L, max_length, device, backbone):
		""" Compute AOPC https://arxiv.org/pdf/1509.06321.pdf"""
		
		
		input_dict ={}
		input_dict['input_ids'] = []
		input_dict['attention_mask'] = []
		prediction_label, prediction_prob = [], []
		for text_no, text in enumerate(sentences):
				words_to_remove = []
				words_to_remove_all = np.argsort(np.abs([float(i) for i in weights[text_no].split(',') ]))[::-1]
				
				text = text.split(',')
				for i, word_index in enumerate(words_to_remove_all):
						if i == L or i>=len(text):
								break
						words_to_remove.append(word_index)
						new_text = [token for i, token in enumerate(tokenizer.convert_tokens_to_ids(text)) if i not in words_to_remove][:max_length - 2]
						
						num_pads = max_length - len(new_text) - 2
						ids = [START[backbone]] + new_text + [END[backbone]] + [PAD[backbone]]*num_pads
						
						attention = [1]*(len(new_text)+2) + [0]*num_pads
						ids = torch.LongTensor(ids).unsqueeze(0).to(device)
						attention = torch.LongTensor(attention).unsqueeze(0).to(device)
						
						prediction_prob.append(pred_probs[text_no])
						prediction_label.append(pred_labels[text_no])
						
						input_dict['input_ids'].append(ids)
						input_dict['attention_mask'].append(attention)
		input_dict['input_ids'] = torch.cat(input_dict['input_ids'], dim = 0)
		input_dict['attention_mask'] = torch.cat(input_dict['attention_mask'], dim = 0)
		

		
		return torch.Tensor(prediction_prob).detach().numpy(), prediction_label, input_dict
def aopc_per_batch(orig_probs, orig_labels, input_dict, model, args, L=10):
		with torch.no_grad():
				changed_probs= torch.softmax( model(input_dict, flag = 'test', interpret_mode = 'lime', mode = args.baseline)[0], dim = -1)[:, orig_labels].detach().cpu().numpy()[0]
				#print(len(orig_labels), orig_probs.shape, changed_probs.shape)
				delta = (orig_probs - changed_probs).sum()
				#quit()
		return delta

def AOPC(all_sentences, weights, all_prediction, all_probs, model, tokenizer, args, L=10):
		start = 0
		end = min(len(all_sentences), args.per_gpu_eval_batch_size)
		delta = 0.0
		print('Total number of interpretations: ', len(all_sentences))
		while end <= len(all_sentences):
				#print(end)
				orig_probs, orig_labels, input_dict = compute_perturbations(all_sentences[start:end], weights[start:end], all_prediction[start:end], all_probs[start:end], tokenizer, L, args.max_sent_len, args.device, args.backbone)
				curr_delta = aopc_per_batch(orig_probs, orig_labels, input_dict, model, args, L)
				#print(curr_delta)
				delta += curr_delta
				start = end
				end += args.per_gpu_eval_batch_size
		return delta / (len(all_sentences)*(L))



