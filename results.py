import torch
from torchvision import datasets 
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from model.LeNet5 import LeNet5
from profiler import profiler
from coverage import Coverage
from utils.augmented import Augmented
from utils.fid import fid

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# import dataset 
train_dataset = datasets.MNIST(root="mnist/", train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='mnist/', train=False, transform=ToTensor())

# create dataloaders
batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# import trained model 
state_dict = torch.load('./model/lenet5.pt')
model = LeNet5().to(device)
model.load_state_dict(state_dict)
model.eval()

# profile model
cov_dict = profiler(model, device, train_dataloader)

# determine coverage
cov = Coverage(model, device, test_dataloader, cov_dict)
nbcov = cov.NBCoverage()
snacov = cov.SNACoverage()

# import diffusion generated images
ddpm_k1 = torch.load('ddpm_corrected_k_1.pt')
ddpm_k2 = torch.load('ddpm_corrected_k_2.pt')
ddim_k1 = torch.load('ddim_corrected_k_1.pt')
ddim_k2 = torch.load('ddim_corrected_k_2.pt')

# covert into torch dataset
ddpm_k1_dataset = test_dataset+Augmented(ddpm_k1)
ddpm_k2_dataset = test_dataset+Augmented(ddpm_k2)
ddim_k1_dataset = test_dataset+Augmented(ddim_k1)
ddim_k2_dataset = test_dataset+Augmented(ddim_k2)

# convert into dataloader
ddpm_k1_dataloader = DataLoader(ddpm_k1_dataset, batch_size=batch_size)
ddpm_k2_dataloader = DataLoader(ddpm_k2_dataset, batch_size=batch_size)
ddim_k1_dataloader = DataLoader(ddim_k1_dataset, batch_size=batch_size)
ddim_k2_dataloader = DataLoader(ddim_k2_dataset, batch_size=batch_size)

# coverage increase results
cov_ddpm_k1 = Coverage(model, device, ddpm_k1_dataloader, cov_dict)
cov_ddpm_k2 = Coverage(model, device, ddpm_k2_dataloader, cov_dict)
cov_ddim_k1 = Coverage(model, device, ddim_k1_dataloader, cov_dict)
cov_ddim_k2 = Coverage(model, device, ddim_k2_dataloader, cov_dict)

nbcov_ddpm_k1 = cov_ddpm_k1.NBCoverage()
snacov_ddpm_k1 = cov_ddpm_k1.SNACoverage()

nbcov_ddpm_k2 = cov_ddpm_k2.NBCoverage()
snacov_ddpm_k2 = cov_ddpm_k2.SNACoverage()

nbcov_ddim_k1 = cov_ddim_k1.NBCoverage()
snacov_ddim_k1 = cov_ddim_k1.SNACoverage()

nbcov_ddim_k2 = cov_ddim_k2.NBCoverage()
snacov_ddim_k2 = cov_ddim_k2.SNACoverage()

# calculate fid scores
ddpm_fid = fid(test_dataset, Augmented(ddpm_k1))
ddim_fid = fid(test_dataset, Augmented(ddim_k1))


