import torch 
import torch.nn as nn
import torch.nn.functional as F
from repr_utils import *
set_seed(1)
from imask import IMASK
import numpy as np
SMALL = 1e-08

class BERT(nn.Module):

        def __init__(self, args, prebert, mapper, co_occur = np.zeros((10,10))):
                super(BERT, self).__init__()


                self.args = args
                self.maskmodel = IMASK(args)
                # prebert is the model pretrained on a dataset
                self.bertmodel = prebert
                
                
                
                #self.dropout = args.dropout
                self.symmetric = args.symmetric

                # learnable interaction matrix initialized to zero 
                self.imask = nn.Parameter(torch.zeros((args.subset, args.subset)))
                self.imask.requires_grad = True
                
                # subset of vocabulary  # V X V'
                self.mapper = nn.Parameter(torch.from_numpy(mapper).float()) # tells up if word index i from old vocab is word index j in new vocab
                self.mapper.requires_grad = False
                # make learnt matrix closer to co_occur matrix  
                self.co_occur_matrix = nn.Parameter(torch.from_numpy(co_occur).float())
                self.co_occur_matrix.requires_grad = False



        def get_embeddings(self, inputs):
            # pass tokenized inputs to model
            
            outputs = self.bertmodel(**inputs, output_hidden_states=True, output_attentions=False)
            
            # loss, logits, hidden_states, attention matrices
            # embeddings outputs[2][0]
            if 'labels' in inputs.keys():
                embeddings = outputs[2][0]
            else:
                embeddings = outputs[1][0]
            assert embeddings.size(2) == self.maskmodel.embed_dim
            
            return embeddings
        def get_backbone_model_outputs(self, inputs):
            # pass tokenized inputs to model
            outputs = self.bertmodel(**inputs, output_hidden_states=True, output_attentions=False)
            # loss, logits, hidden_states, attention matrices
            # embeddings outputs[2][0]
            loss = outputs[0]
            logits = outputs[1]
            
            return loss, logits
            

        def extract_sentence_level_mask(self, x):
            extracted_mask = []
            for seq in range(x.size(0)):
                # L X V'
                temp_x = self.mapper[x[seq].squeeze(),:].to(x.device) # the multiplication is between {L X V'} AND { (L X V') (V' X V') }' -- 1 X L X L
                # get subsampled local mask
                extracted_mask.append(torch.matmul(temp_x, torch.matmul(temp_x, self.imask).t()).unsqueeze(0))
                
            
            extracted_mask = (torch.cat((extracted_mask), dim=0)).unsqueeze(3)
            return extracted_mask

        def forward(self, inputs, flag, mode, interpret_mode = 'prediction'):
            
            
            if self.symmetric: self.maskmodel.imask.data = (self.model.imask.data + torch.transpose(self.model.imask.data, 0, 1))/2.0
            if self.args.finetune:
                outputs = self.bertmodel(**inputs, output_hidden_states=True, output_attentions=False)
                if interpret_mode == 'lime':
                    if 'labels' in inputs.keys():
                        pooled_output = outputs[2][-1][:,0,...]
                        return outputs[1], pooled_output
                    else:
                        pooled_output = outputs[1][-1][:,0,...]
                        return outputs[0], pooled_output

                return outputs[1], outputs[0], None, None, None, None, None
            embeddings = self.get_embeddings(inputs) # embeddings are not updated if static
            
            extracted_mask = self.extract_sentence_level_mask(inputs['input_ids'])     
            
            extracted_mask = torch.sigmoid(extracted_mask)
            #print(extracted_mask)
            sparse_loss = torch.sum((extracted_mask).squeeze(3))# sparse loss is sum of the probabilities  
            
            graph_loss = torch.sum(extracted_mask * torch.log(extracted_mask + SMALL) + (1.0 - extracted_mask) * torch.log((1.0 - extracted_mask) + SMALL))
            
            extracted_mask_0 = 1.0 - extracted_mask # get probs for not-an-interaction
            extracted_mask	= torch.cat((extracted_mask_0, extracted_mask), dim=3)
            
            if self.args.baseline and self.args.beta_i == 0.0:
                x_prime, vmask = self.maskmodel(embeddings, extracted_mask, flag,  baseline = mode)
                x_prime = embeddings
            else:
                x_prime, vmask = self.maskmodel(embeddings, extracted_mask, flag,  baseline = mode)

            
            if 'labels' in inputs.keys():
                output = self.bertmodel(inputs_embeds = x_prime, labels = inputs['labels'], output_hidden_states=True, output_attentions=False)
            else:
                output = self.bertmodel(inputs_embeds = x_prime,  output_hidden_states=True, output_attentions=False)
            if interpret_mode == 'lime':
                if 'labels' in inputs.keys():
                    pooled_output = output[2][-1][:,0,...]
                    return output[1], pooled_output
                else:
                    pooled_output = output[1][-1][:,0,...]
                    return output[0], pooled_output



            probs_pos = F.softmax(vmask,dim=2)[:,:,1]
            probs_neg = F.softmax(vmask,dim=2)[:,:,0]
            infor_loss = torch.sum(probs_pos * torch.log(probs_pos+SMALL) + probs_neg*torch.log(probs_neg+SMALL)) 

            # prediction_logits, prediction_loss, infor_loss, graph_loss, sparse_loss, local_graphs, local_words
            return output[1], output[0], infor_loss/(2*((extracted_mask[...,1].size(1)))), graph_loss/(2*((extracted_mask[...,1].size(1))**2)), sparse_loss/(2*((extracted_mask[...,1].size(1))**2)) , extracted_mask[...,1], probs_pos
                

                    
