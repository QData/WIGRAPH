

class params:

    def __init__(self, args):
        # params that are hyperparameters

        self.opts = vars(args)
        self.variable_keys  = ['backbone_drop','interaction_embeddings', 'anneal', 'learning_rate', 'max_sent_len', 'init_mode','model_name_or_path', 'backbone', 'direct_base_model_training', 'test_only','task_name', 'beta_i', 'beta_g', 'beta_s', 'factor', 'baseline', 'imask_dropout', 'mask_hidden_dim', 'seed', 'onlyA', 'non_linearity']
        


    def save_string(self):
        return_str = ''
        return_str += self.opts['task_name'] + '/'
        for i,key in enumerate(self.variable_keys):
            if i !=0:
                    return_str += '_'
            if key != 'task_name':
                
                return_str += key + '_' + str(self.opts[key])
            if i%4 == 0 and i!=0:
                return_str += '/'


        return return_str

    def get_dict_obj(self):
        saved_obj = []
        for key in self.opts.keys():
            saved_obj .append( key + ':' + str(self.opts[key]) + '\n')



        return saved_obj

