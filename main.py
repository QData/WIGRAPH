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
from DATA_CONFIGS import *
from mask_model import *
import numpy as np
import pandas as pd



from mask_model import *    


def main():
    
    args  = get_args()
    set_seed(args.seed)
    if args.task_name == 'imdb': 
        args.max_sent_len = 512
        args.per_gpu_train_batch_size = 16
        args.per_gpu_eval_batch_size = 32

    # load pretrained model 
    if args.backbone == 'bert':
            args.model_name_or_path = 'bert-base-uncased'
    if args.backbone == 'roberta':
            args.model_name_or_path = 'roberta-base'
    if args.backbone == 'distilbert':
            args.model_name_or_path = 'distilbert-base-uncased'
    if args.direct_base_model_training:
        config = AutoConfig.from_pretrained(args.model_name_or_path) 
        config.hidden_dropout_prob = args.backbone_drop
        bert_backbone = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels = args.class_num)
    else:
        # load model
        bert_backbone = torch.load('finetuned_models/'+args.backbone+'/'+args.task_name+'/best_model.pt').bertmodel

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    
    
    
    
    
    word_freqs = np.load('metadata/'+args.task_name.lower()+ '_'+args.model_name_or_path.replace('/', '_')+'_word_freqs_counts_2.npy')
    mapper = np.zeros((tokenizer.vocab_size, args.subset))
    
    assert word_freqs.shape == (tokenizer.vocab_size,)
    include_word_idx = np.argsort(word_freqs)[-1*int(args.subset):]

    for mapper_idx,i in enumerate(include_word_idx):
            if i not in SPECIAL_TOKENS[args.backbone]:
                mapper[i, mapper_idx] = 1.0
    
    
    
    dataset_splits = get_dataset(args.task_name)
    train_dataset = DataLoader( TextDataset(dataset = dataset_splits['train'], tokenizer = tokenizer, length = args.max_sent_len, device = args.device) , batch_size = args.per_gpu_train_batch_size, shuffle = True)
    valid_dataset = DataLoader( TextDataset(dataset = dataset_splits['valid'], tokenizer =  tokenizer, length = args.max_sent_len, device = args.device) , batch_size = args.per_gpu_eval_batch_size, shuffle = False)
    test_dataset = DataLoader( TextDataset(dataset = dataset_splits['test'], tokenizer = tokenizer, length = args.max_sent_len, device = args.device) , batch_size = args.per_gpu_eval_batch_size, shuffle = False)
    model = BERT(args, bert_backbone, mapper).to(args.device)
    
    parametersInit = filter(lambda p: p.requires_grad, model.bertmodel.parameters())
    for param in parametersInit:
        param.requires_grad = True
    # fix embeddings
    if args.init_mode == 'static':
        if args.backbone == 'bert':
            parameters = filter(lambda p: p.requires_grad, model.bertmodel.bert.embeddings.parameters())
        if args.backbone == 'roberta':
            parameters = filter(lambda p: p.requires_grad, model.bertmodel.roberta.embeddings.parameters())
        if args.backbone == 'distilbert':
            parameters = filter(lambda p: p.requires_grad, model.bertmodel.distilbert.embeddings.parameters())
        for param in parameters:
            param.requires_grad = False
    
    
    print("starting model training!")
    train_model(model, train_dataset, valid_dataset, test_dataset, args)



    
if __name__ == "__main__":
    main()



