import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from collections import OrderedDict
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model.LeNet5 import LeNet5

def profiler(model, device, dataloader):
    model.to(device)
    model.eval()
    cov_dict = OrderedDict()
    nodes = get_graph_node_names(model)[0]
    nodes.pop(0)
    feature_extractor = create_feature_extractor(model, return_nodes=nodes)

    for idx, (data, label) in tqdm(enumerate(dataloader)):
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
                    if (name, neuron_id) not in cov_dict:
                        cov_dict[(name, neuron_id)] = [None, None]
                    
                    profile_data_list = cov_dict[(name, neuron_id)]

                    lower_bound = neurons_min.values[neuron_id]
                    upper_bound = neurons_max.values[neuron_id]
                    if profile_data_list != [None, None]:
                        if upper_bound < profile_data_list[1]:
                            upper_bound = profile_data_list[1]
                            
                        if lower_bound > profile_data_list[0]:
                            lower_bound = profile_data_list[0]
                    
                    profile_data_list[0] = lower_bound
                    profile_data_list[1] = upper_bound 
                    cov_dict[(name, neuron_id)] = profile_data_list
    
    return cov_dict

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    # import datasets
    train_dataset = datasets.MNIST(root="mnist/", train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='mnist/', train=False, transform=ToTensor())

    # create dataloaders
    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # load in model
    model = LeNet5().to(device)
    state_dict = torch.load('./model/lenet5.pt')
    model.load_state_dict(state_dict)
    model.eval()

    # profile model and data
    cov_dict = profiler(model, device, train_dataloader)
