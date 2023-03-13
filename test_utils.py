import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset
from logger_utils import stats
from tqdm import tqdm
def test_epoch(model, dataset, args):
    predictions_epoch, label_array, max_graph_epoch , min_graph_epoch = [], [], [] , []
    total_loss_epoch, prediction_loss_epoch, sparse_loss_epoch, infor_loss_epoch, graph_loss_epoch, sparse_loss_epoch = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    model.eval()
    
    with torch.no_grad():
        for bz_number, batch in tqdm(enumerate(dataset)):
            batch_size = batch['input_ids'].size(0)
            label_array.extend( batch['labels'].squeeze(1).cpu().numpy())
            
            prediction_logits, prediction_loss, infor_loss, graph_loss, sparse_loss, local_graphs, local_words = model(batch, 
                                                                                                flag = 'test', mode =args.baseline)
            
            
            predictions = prediction_logits.max(dim = -1).indices
            prediction_loss_epoch += ( prediction_loss.item() )
            
            if not args.finetune:
                max_graph_epoch.append(local_graphs.max().item())
                min_graph_epoch.append(local_graphs.min().item())
            
                
                sparse_loss_epoch += ( args.beta_s * sparse_loss.item() )
                infor_loss_epoch += ( args.beta_i * infor_loss.item() )
                graph_loss_epoch += ( args.beta_g * graph_loss.item() )
                total_loss = args.beta_pred * prediction_loss + args.beta_i * infor_loss + args.beta_g * graph_loss + args.beta_s * sparse_loss
            else:
                total_loss = prediction_loss

            

            
            predictions_epoch.extend(predictions.cpu().numpy())

            total_loss /= batch_size
            
            
            total_loss_epoch += (total_loss.item())
        
        
        

    current_acc = sum(p == l for p,l in zip(predictions_epoch, label_array))/ len(label_array)
    
    #quit()
    if not args.finetune:
        return  current_acc, predictions_epoch, label_array, total_loss_epoch/len(label_array), \
            prediction_loss_epoch/len(label_array), sparse_loss_epoch/len(label_array), \
            infor_loss_epoch/len(label_array), graph_loss_epoch/len(label_array),max(max_graph_epoch),min(min_graph_epoch)
    else:
        return  current_acc, predictions_epoch, label_array, total_loss_epoch/len(label_array), \
            prediction_loss_epoch/len(label_array), 0.0, \
            0.0, 0.0 ,0.0, 0.0
