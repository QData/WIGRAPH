
class stats:
    def __init__(self, keys, save_dir):
        keys += ['max_interaction', 'min_interaction']
        self.stats_train_dict = {}
        self.current_best = -1.0
        self.current_test_best = -1.0
        self.current_graph_highest = -1.0
        self.current_graph_lowest = -1.0
        self.save_dir = save_dir
        self.best_epoch = 0

        for key in keys:
            self.stats_train_dict[key] = []
        self.stats_valid_dict = {}
        for key in keys:
            self.stats_valid_dict[key] = []
        self.stats_test_dict = {}
        for key in keys:
            self.stats_test_dict[key] = []
    
    

    def best(self):
        if self.stats_valid_dict['accuracy'][-1] >= self.current_best:
            self.current_best = self.stats_valid_dict['accuracy'][-1]
            self.current_test_best = self.stats_test_dict['accuracy'][-1]
            self.best_epoch = len(self.stats_test_dict['accuracy']) - 1
            return True
        else:
            return False
    def update(self, mode, current_acc, total_loss, prediction_loss, sparse_loss, infor_loss, graph_loss, max_graph, min_graph):
        if mode == 'train':
            self.stats_train_dict['accuracy'].append(current_acc)
            self.stats_train_dict['total_loss'].append(total_loss)
            self.stats_train_dict['prediction_loss'].append(prediction_loss)
            self.stats_train_dict['infor_loss'].append(infor_loss)
            self.stats_train_dict['graph_loss'].append(graph_loss)
            self.stats_train_dict['sparse_loss'].append(sparse_loss)
            self.stats_train_dict['max_interaction'].append(max_graph)
            self.stats_train_dict['min_interaction'].append(min_graph)
        elif mode == 'valid':
            self.stats_valid_dict['accuracy'].append(current_acc)
            self.stats_valid_dict['total_loss'].append(total_loss)
            self.stats_valid_dict['prediction_loss'].append(prediction_loss)
            self.stats_valid_dict['infor_loss'].append(infor_loss)
            self.stats_valid_dict['graph_loss'].append(graph_loss)
            self.stats_valid_dict['sparse_loss'].append(sparse_loss)
            self.stats_valid_dict['max_interaction'].append(max_graph)
            self.stats_valid_dict['min_interaction'].append(min_graph)
        else:
            self.stats_test_dict['accuracy'].append(current_acc)
            self.stats_test_dict['total_loss'].append(total_loss)
            self.stats_test_dict['prediction_loss'].append(prediction_loss)
            self.stats_test_dict['infor_loss'].append(infor_loss)
            self.stats_test_dict['graph_loss'].append(graph_loss)
            self.stats_test_dict['sparse_loss'].append(sparse_loss)
            self.stats_test_dict['max_interaction'].append(max_graph)
            self.stats_test_dict['min_interaction'].append(min_graph)
  
