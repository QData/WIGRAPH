import os
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset
from logger_utils import stats
from tqdm import tqdm
from test_utils import *

def train_epoch(model, dataset, args, optimizer):
    predictions_epoch, label_array, max_graph_epoch , min_graph_epoch = [], [], [] , []
    total_loss_epoch, prediction_loss_epoch, sparse_loss_epoch, infor_loss_epoch, graph_loss_epoch, sparse_loss_epoch = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    model.train()
    for bz_number, batch in tqdm(enumerate(dataset)):
        batch_size = batch['input_ids'].size(0)
        
        model.zero_grad()
        optimizer.zero_grad()
        label_array.extend( batch['labels'].squeeze(1).cpu().numpy())
        
        prediction_logits, prediction_loss, infor_loss, graph_loss, sparse_loss, local_graphs, local_words = model(batch, 
                                                                                            flag = 'train', mode =args.baseline)
        
        
        
        
        predictions = prediction_logits.max(dim = -1).indices
        prediction_loss_epoch += ( prediction_loss.item() )
        #print(local_graphs.max())
        if not args.finetune:
            max_graph_epoch.append(local_graphs.max().item())
            min_graph_epoch.append(local_graphs.min().item())
        
            
            sparse_loss_epoch += ( args.beta_s*sparse_loss.item() )
            infor_loss_epoch += ( args.beta_i*infor_loss.item() )
            graph_loss_epoch += ( args.beta_g*graph_loss.item() )
            total_loss = args.beta_pred * prediction_loss + args.beta_i * infor_loss + args.beta_g * graph_loss + args.beta_s * sparse_loss
        else:
            total_loss = prediction_loss

        

        
        predictions_epoch.extend(predictions.cpu().numpy())

        total_loss /= batch_size
        total_loss.backward()
        optimizer.step()
        
        total_loss_epoch += (total_loss.item())
        
        
        

    current_acc = sum(p == l for p,l in zip(predictions_epoch, label_array))/ len(label_array)
    if not args.finetune:
        return  current_acc, predictions_epoch, label_array, total_loss_epoch/len(label_array), \
            prediction_loss_epoch/len(label_array), sparse_loss_epoch/len(label_array), \
            infor_loss_epoch/len(label_array), graph_loss_epoch/len(label_array),max(max_graph_epoch),min(min_graph_epoch)
    else:
        return  current_acc, predictions_epoch, label_array, total_loss_epoch/len(label_array), \
            prediction_loss_epoch/len(label_array), 0.0, \
            0.0, 0.0 ,0.0, 0.0




  
        

def train_model(model, train_dataset, valid_dataset, test_dataset, args):

    

    # Prepare optimizer and schedule (linear warmup and decay)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n != 'imask'], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n != 'imask'], 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if n == 'imask'], 'lr':args.factor*args.learning_rate ,  'weight_decay': args.weight_decay}
                ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    
    model_stats = stats(['accuracy',  'total_loss', 'prediction_loss', 'infor_loss', 'graph_loss', 'sparse_loss'], save_dir = args.save_dir)
    if not args.test_only:
        epoch_train_acc, _, _, total_loss_train_epoch, prediction_loss_train_epoch, \
                        sparse_loss_train_epoch, infor_loss_train_epoch, graph_loss_train_epoch,max_graph_train, min_graph_train = test_epoch(model, train_dataset, args)
        model_stats.update('train', epoch_train_acc, total_loss_train_epoch, prediction_loss_train_epoch, sparse_loss_train_epoch, infor_loss_train_epoch, graph_loss_train_epoch, max_graph_train, min_graph_train)

        epoch_valid_acc, _, _, total_loss_valid_epoch, prediction_loss_valid_epoch, \
                        sparse_loss_valid_epoch, infor_loss_valid_epoch, graph_loss_valid_epoch, max_graph_valid, min_graph_valid = test_epoch(model, valid_dataset, args)
        model_stats.update('valid', epoch_valid_acc, total_loss_valid_epoch, prediction_loss_valid_epoch, sparse_loss_valid_epoch, infor_loss_valid_epoch, graph_loss_valid_epoch, max_graph_valid, min_graph_valid )

        epoch_test_acc, _, _, total_loss_test_epoch, prediction_loss_test_epoch, \
                        sparse_loss_test_epoch, infor_loss_test_epoch, graph_loss_test_epoch, max_graph_test, min_graph_test = test_epoch(model, test_dataset, args)
        model_stats.update('test', epoch_test_acc, total_loss_test_epoch, prediction_loss_test_epoch, sparse_loss_test_epoch, infor_loss_test_epoch, graph_loss_test_epoch, max_graph_test, min_graph_test)
        
        for epoch in tqdm(range(1, args.num_train_epochs+1)):
            print('beta values: ', args.beta_i, args.beta_s, args.beta_g)
            epoch_train_acc, _, _, total_loss_train_epoch, prediction_loss_train_epoch, \
                        sparse_loss_train_epoch, infor_loss_train_epoch, graph_loss_train_epoch,max_graph_train, min_graph_train = train_epoch(model, train_dataset, args, optimizer)
            model_stats.update('train', epoch_train_acc, total_loss_train_epoch, prediction_loss_train_epoch, sparse_loss_train_epoch, infor_loss_train_epoch, graph_loss_train_epoch, max_graph_train, min_graph_train)

            epoch_valid_acc, _, _, total_loss_valid_epoch, prediction_loss_valid_epoch, \
                        sparse_loss_valid_epoch, infor_loss_valid_epoch, graph_loss_valid_epoch, max_graph_valid, min_graph_valid = test_epoch(model, valid_dataset, args)
            model_stats.update('valid', epoch_valid_acc, total_loss_valid_epoch, prediction_loss_valid_epoch, sparse_loss_valid_epoch, infor_loss_valid_epoch, graph_loss_valid_epoch, max_graph_valid, min_graph_valid )

            epoch_test_acc, _, _, total_loss_test_epoch, prediction_loss_test_epoch, \
                        sparse_loss_test_epoch, infor_loss_test_epoch, graph_loss_test_epoch, max_graph_test, min_graph_test = test_epoch(model, test_dataset, args)
            model_stats.update('test', epoch_test_acc, total_loss_test_epoch, prediction_loss_test_epoch, sparse_loss_test_epoch, infor_loss_test_epoch, graph_loss_test_epoch, max_graph_test, min_graph_test)

            model_name = args.save_name.replace('test_only_False_', '') 
            os.makedirs(os.path.join(args.save_dir, model_name), exist_ok = True)
            if model_stats.best():
                print('found new model!')
                if True:
                    torch.save(model.cpu(), os.path.join(args.save_dir, model_name , 'best_model.pt'))
                    model = model.to(args.device)
                    
            
            model = model.to(args.device)
            
            if args.anneal:
                if args.beta_i > 0.01: 
                    args.beta_i -= 0.099
                if args.beta_g > 0.01: 
                    args.beta_g -= 0.099
                if args.beta_s > 0.01: 
                    args.beta_s -= 0.099

    else:
            model_name = args.save_name.replace('test_only_True_', '')
            model = (torch.load(os.path.join(args.save_dir,model_name , 'best_model.pt'))).to(args.device)
            
            epoch_train_acc, _, _, total_loss_train_epoch, prediction_loss_train_epoch, \
                        sparse_loss_train_epoch, infor_loss_train_epoch, graph_loss_train_epoch,max_graph_train, min_graph_train = test_epoch(model, train_dataset, args)
            model_stats.update('train', epoch_train_acc, total_loss_train_epoch, prediction_loss_train_epoch, sparse_loss_train_epoch, infor_loss_train_epoch, graph_loss_train_epoch, max_graph_train, min_graph_train)

            epoch_valid_acc, _, _, total_loss_valid_epoch, prediction_loss_valid_epoch, \
                        sparse_loss_valid_epoch, infor_loss_valid_epoch, graph_loss_valid_epoch, max_graph_valid, min_graph_valid = test_epoch(model, valid_dataset, args)
            model_stats.update('valid', epoch_valid_acc, total_loss_valid_epoch, prediction_loss_valid_epoch, sparse_loss_valid_epoch, infor_loss_valid_epoch, graph_loss_valid_epoch, max_graph_valid, min_graph_valid )

            epoch_test_acc, _, _, total_loss_test_epoch, prediction_loss_test_epoch, \
                        sparse_loss_test_epoch, infor_loss_test_epoch, graph_loss_test_epoch, max_graph_test, min_graph_test = test_epoch(model, test_dataset, args)
            model_stats.update('test', epoch_test_acc, total_loss_test_epoch, prediction_loss_test_epoch, sparse_loss_test_epoch, infor_loss_test_epoch, graph_loss_test_epoch, max_graph_test, min_graph_test)
        
            


            

