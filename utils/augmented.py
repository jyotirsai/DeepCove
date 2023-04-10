from torch.utils.data import Dataset

class Augmented(Dataset):
  def __init__(self, tests):
    self.samples = tests
  
  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, idx):
    return self.samples[idx]