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


def get_orig_model_outputs(inputs, model, mode):
  orig_output, orig_embedding = model(inputs, flag = 'test', mode = mode, interpret_mode='lime')
  orig_output = orig_output.detach()
  orig_embedding = orig_embedding.detach()
  return orig_output, orig_embedding
  
def shap_per_example(inputs, model, args, tokenizer):
    '''
    x : sentence to be interpreted 
    
    '''
    with torch.no_grad():
      orig_output, orig_embedding = get_orig_model_outputs(inputs, model, args.baseline)
    pred_label = torch.max(orig_output, dim = -1)[1].item()
    assert inputs['input_ids'].size(0) == 1
    feature_indices = [idx for idx,tk in enumerate(inputs['input_ids'].squeeze().tolist()) if tk not in SPECIAL_TOKENS[args.backbone]]
    feature_importance = []
    for idx in feature_indices:
      batch_1 = {}
      batch_1['input_ids'] = copy.deepcopy( inputs['input_ids'] ).repeat(args.num_perturb,  1)
      batch_1['attention_mask'] = copy.deepcopy(inputs['attention_mask'].repeat(args.num_perturb,  1))
      
      random_length = np.random.randint(1, len(feature_indices), args.num_perturb)
      for i in range(args.num_perturb):
        inactive_tokens = random.sample(feature_indices, random_length[i])
        batch_1['input_ids'][i, inactive_tokens] = 0
      batch_2 = copy.deepcopy(batch_1)
      #print(batch_2['input_ids'].size())
      batch_1['input_ids'][:, idx] = inputs['input_ids'].squeeze().tolist()[idx]
      batch_2['input_ids'][:, idx] = 0
      
      with torch.no_grad():
        feature_output = torch.softmax(get_orig_model_outputs(batch_1, model, mode = args.baseline)[0], dim = -1)
        inactive_output = torch.softmax( get_orig_model_outputs(batch_2, model, mode = args.baseline)[0], dim = -1)
        
        feature_importance.append( (feature_output[:, pred_label] - inactive_output[:, pred_label]).sum().item()/ args.num_perturb)



      

    tokens  = [tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist()[c]) for c in feature_indices]
    return tokens, feature_importance, feature_importance, np.argmax(orig_output.detach().cpu().numpy(), axis=1), torch.softmax(orig_output.detach().cpu(), dim = -1)[:,np.argmax(orig_output.detach().cpu().numpy(), axis=1)].numpy()[0][0]
    
        
        
def interpret_with_shap(test_dataset, labels, model, tokenizer, args ):
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
      inputs = tokenizer(sentence, padding='max_length', truncation=True, return_tensors="pt", max_length = 128)
      data_sample =  {
                  'input_ids': inputs['input_ids'].to(args.device),
                  'attention_mask': inputs['attention_mask'].to(args.device),
                  'labels': torch.Tensor([label]).long().to(args.device),
                  }
    
      sample_no += 1
      mod_sentence, word_score, x_prob, pred, pred_prob = shap_per_example(data_sample, model, args, tokenizer)
      end = time.time()
      with open(save_interpretation_file, 'a') as f:
      
        f.write(sentence + ' >> ' + ','.join([word for word in mod_sentence]) \
          + ' >> ' + ','.join([str(w) for w in word_score]) \
            +' >> ' + ','.join([str(w) for w in x_prob]) \
            + ' >> ' + str(pred_prob)  + ' >> ' + str(label) + ' >> ' + str(pred[0]) \
                + ' >> ' + str(end - start) + '\n')
      print('time for one sample of length ', len(sentence.split()),' =  ', end -  start)
      if sample_no == args.num_examples:
        quit()

    



          


            



