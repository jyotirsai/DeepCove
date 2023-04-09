import torch 
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from collections import OrderedDict
from tqdm.auto import tqdm

class Coverage:
    def __init__(self, model, device, dataloader, cov_dict):
        self.model = model 
        model.to(device)
        self.model.eval()
        self.dataloader = dataloader 
        self.cov_dict = cov_dict
        self.CoverageRecorder()
    
    def CoverageRecorder(self):
        self.coverage_recorder = OrderedDict()
        nodes = get_graph_node_names(self.model)[0]
        nodes.pop(0)
        feature_extractor = create_feature_extractor(self.model, return_nodes=nodes)
        self.upper_neurons = OrderedDict()

        for idx, (data, label) in tqdm(enumerate(self.dataloader)):
            with torch.no_grad():
                out_list = feature_extractor(data)

                for name, layer in out_list.items():
                    cur_neuron_num = 1
                    if len(layer.shape) == 2:
                        cur_neuron_num = layer.shape[1]
                    else:
                        cur_neuron_num = layer.shape[1]*layer.shape[2]*layer.shape[3]
                    
                    layer = torch.flatten(layer, start_dim=1)
                    neurons_max = torch.max(layer, dim=0)
                    neurons_min = torch.min(layer, dim=0)

                    for neuron_id in range(cur_neuron_num):
                        if (name, neuron_id) not in self.coverage_recorder:
                            self.coverage_recorder[(name, neuron_id)] = [0, 0]
                        
                        if neurons_min.values[neuron_id] < self.cov_dict[(name, neuron_id)][0]:
                            self.coverage_recorder[(name, neuron_id)][0] = 1
                        
                        if neurons_max.values[neuron_id] > self.cov_dict[(name, neuron_id)][1]:
                            self.coverage_recorder[(name, neuron_id)][1] = 1
                            self.upper_neurons[(name, neuron_id)][1] = 1
    
    def NBCoverage(self):
        return sum(sum(i) for i in self.coverage_recorder.values()) / (2*len(self.coverage_recorder.values()))

    def SNACoverage(self):
        return sum(sum(i) for i in self.upper_neurons.values()) / (len(self.coverage_recorder.values()))


