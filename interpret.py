
import torch
from repr_utils import *
set_seed(1)
from args_utils import *
from train_utils import *
from test_utils import *
from logger_utils import *
from local_dataset import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset
from local_dataset import TextDataset

from mask_model import *
import numpy as np
import wandb
import pandas as pd
from lime.lime_utils import *
from shap.shap_utils import *


from mask_model import *    
from aopc_utils import *

def main():
    args  = get_args()
    set_seed(args.seed)
    
    # load pretrained model on  dataset
    if args.backbone == 'bert':
            args.model_name_or_path = 'bert-base-uncased'
    if args.backbone == 'roberta':
            args.model_name_or_path = 'roberta-base'
    if args.backbone == 'distilbert':
            args.model_name_or_path = 'distilbert-base-uncased'
    if args.direct_base_model_training:
        config = AutoConfig.from_pretrained(args.model_name_or_path) 
        config.hidden_dropout_prob = args.backbone_drop
        bert_backbone = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config = config, num_labels = args.class_num)
    else:
        # load model

        bert_backbone = torch.load('finetuned_models/'+args.backbone+'/'+args.task_name+'/best_model.pt').bertmodel

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
  
    dataset_splits = get_dataset(args.task_name)
    test_dataset = dataset_splits['test']
    shuffled_indices = [i for i in range(len(test_dataset))]
    random.shuffle(shuffled_indices)
    test_dataset_sentences = [test_dataset['sentence'][x] for x in shuffled_indices]
    test_dataset_labels = [test_dataset['label'][x] for x in shuffled_indices]
    
    if not args.finetune:
        model_name = args.save_name.replace('test_only_False_', '')
        model = (torch.load(os.path.join(args.save_dir,model_name , 'best_model.pt'))).to(args.device)
    else:
        model = torch.load('finetuned_models/'+args.backbone+'/'+args.task_name+'/best_model.pt').to(args.device)
        model_name = args.backbone+'_'+args.task_name
        os.makedirs(os.path.join(args.save_dir, model_name), exist_ok = True)
    args.interpret_file = os.path.join(args.save_dir,model_name , args.interpret_method+'_interpret_'+str(args.num_perturb)+'.txt')
    
    
    model.eval()
    if args.interpret_method == 'lime':
        interpret_with_lime(test_dataset_sentences, test_dataset_labels, model, tokenizer, args )
    else:
        interpret_with_shap(test_dataset_sentences, test_dataset_labels, model, tokenizer, args)
      
    save_interpretation_file = args.interpret_file
    interpret_file = pd.read_csv(save_interpretation_file, sep=' >> ')
    sentences = interpret_file['tokenized_sentences'].tolist()
    weights = interpret_file['linear_weights'].tolist()
    pred_labels = interpret_file['pred_label'].values
    pred_probs = interpret_file['pred_probability'].values
    #print(len(sentences))
    #quit()
    aopc = ( AOPC(sentences, weights , pred_labels, pred_probs, model, tokenizer, args)*100.0)  
    print(aopc)

if __name__ == "__main__":
    main()



