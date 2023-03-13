import argparse
from DATA_CONFIGS import NUM_CLASSES
import datetime
from params import params
import os
import torch

def get_args():
        parser = argparse.ArgumentParser()

        ## Required parameters

        ## imask parameters
        parser.add_argument('--imask-dropout', type=float, default=0.2, help='the probability for dropout')
        parser.add_argument('--subset', type=int, default=10000, help='downsampling the vocabulary')
        parser.add_argument('--imask-layers', type=int, default=1, help='number of hidden layers in imask ')
        parser.add_argument('--mask-hidden-dim', type=int, default=300, help='number of hidden dimension')
        parser.add_argument('--tau', type=float, default=1.0, help='gumbel softmax weighting')
        parser.add_argument('--weighted', action='store_true', help='whether to use W in GCN or not ')
        parser.add_argument('--symmetric', action='store_true', help='whether to treat graph as undirected vs directed')
        parser.add_argument('--baseline', action='store_true', help='whether to only compute vmask')
        parser.add_argument('--anneal', action='store_true', help='whether to reduce beta values')
        parser.add_argument('--finetune', action='store_true', help='finetune base models')
        parser.add_argument('--no-save', action='store_true', help='do not save models')
        parser.add_argument('--embed-dim', type=int, default=768, help='original number of embedding dimension')
        

        ## loss parameters
        parser.add_argument('--beta-i', type=float, default=1, help='information loss for vmask ')
        parser.add_argument('--beta-g', type=float, default=1.0, help='weighting for interaction term ')
        parser.add_argument('--beta-pred', type=float, default=1.0, help='weighting for prediction loss: only use this for debugging purposes ')
        parser.add_argument('--beta-s', type=float, default=0.0, help='weight for sparsity of trainable learnt interaction matrix')
        
        ## bert parameters
        #parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
        parser.add_argument('--max_sent_len', type=int, dest="max_sent_len", default=128, help='max sentence length')
        
        ## training parameters
        parser.add_argument('--factor', type=float, default=1.0, help='factor for separate learning rate for imask: imask will be trained with factor*learning_rate')
        parser.add_argument('--anneal-epoch', type=int, default=100, help='beta')
        parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                                                help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                                                help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--num-perturb", default=100, type=int,
                                                help="for interpretation")                                       
        parser.add_argument("--num-examples", default=500, type=int,
                                                help="number of examples to sample for interpretation.")                                       
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                                help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--learning_rate", default=1e-5, type=float,
                                                help="The initial learning rate for Adam.")
        parser.add_argument("--backbone-drop", default=0.0, type=float,
                                                help="The initial learning rate for Adam.")
        parser.add_argument("--weight-decay", default=0.0, type=float,
                                                help="Weight decay if we apply some.")
        parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                                                help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=10.0, type=float,
                                                help="Max gradient norm.")
        parser.add_argument("--num_train_epochs", default=10, type=int,
                                                help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                                                help="Linear warmup over warmup_steps.")
        parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
        parser.add_argument('--gpu-id', default='0', type=str, help='gpu id')
        parser.add_argument('--non-linearity', default='relu', type=str, help='gpu id')
        

        ## logging parameters
        parser.add_argument('--test-only', action='store_true', help='no training but only test')
        parser.add_argument('--interaction-embeddings', action='store_true', help='whether to start a new set of training runs')
        parser.add_argument('--onlyA', action='store_true', help='do not train R')
        
        parser.add_argument('--new-set', action='store_true', help='whether to start a new set of training runs')
        parser.add_argument('--direct-base-model-training', action='store_true', help='whether to start a new set of training runs')
        
        
        parser.add_argument('--save-dir', type=str, default='WIGRAPH_MODELS/', help='where to save the best model')
        
        

        parser.add_argument('--init-mode', type=str, default='static', help='available models: static, non-static')
        parser.add_argument("--model_name_or_path", default='', type=str,
                                                help="Path to pre-trained model or shortcut name selected in the list: ")
        parser.add_argument("--task-name", default='sst2', type=str,
                                                help="The name of the task to train selected in the list: ")
        parser.add_argument("--interpret-method", default='lime', type=str,
                                                help="The name of the task to train selected in the list: ")
        parser.add_argument("--backbone", default='distilbert', type=str,
                                                help="The name of the backbone model  selected in the list: bert/roberta/distilbert")
        parser.add_argument('--save-name', type=str, default='',
                                                help='path suffix optional to save the final model')
        parser.add_argument('--pretrained-model-path', type=str, default='',
                                                help='path suffix optional to save the final model')
        
        parser.add_argument('--project', type=str, default='WIGRAPH',
                                                help='path suffix optional to save the final model')

        ## Other reproducibility parameters
        parser.add_argument('--seed', type=int, default=42,
                                                help="random seed for initialization")
        args = parser.parse_args()
        
        

        if args.gpu != -1:
                args.device = torch.device("cuda")
        else:
                args.device = torch.device("cpu")
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        
        # Prepare GLUE task
        args.task_name = args.task_name.lower()
        args.class_num = NUM_CLASSES[args.task_name]
        if args.baseline:
                args.beta_s = 0.0
                args.beta_g = 0.0
        if args.onlyA:
                args.beta_i = 0.0
                

        


        variable_params = params(args)
        variable_params_string = variable_params.save_string()
        
        os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok = True)
        print('saving in: ', os.path.join(args.save_dir, args.save_name))
        

        with open(os.path.join(args.save_dir, args.save_name, 'configs.txt'), 'w') as f:
            for config in variable_params.get_dict_obj():

                f.write(config)
        
        

        args.params = variable_params
        return args
    
