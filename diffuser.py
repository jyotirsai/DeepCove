import torch
import torchvision
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel 
from matplotlib import pyplot as plt 
from tqdm import tqdm
from torch.utils.data import DataLoader

# define class-conditioned u-net model from huggingface
class ClassConditionedUNet(nn.module):
    def __init__(self, num_classes, class_emb_size, sample_size, c_in, c_out):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DModel(
            sample_size=sample_size,          
            in_channels=c_in + class_emb_size, 
            out_channels=c_out,           
            layers_per_block=2,      
            block_out_channels=(32, 64, 64), 
            down_block_types=( 
                "DownBlock2D",       
                "AttnDownBlock2D",    
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",      
                "UpBlock2D",         
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape
        class_cond = self.class_emb(class_labels) 
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        net_input = torch.cat((x, class_cond), 1) 
        return self.model(net_input, t).sample

# define diffusion class from huggingface
class Diffuser():
  def __init__(self, device, dataloader, scheduler='ddpm', num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', num_classes=10, class_emb_size=4, sample_size=28, c_in=1, c_out=1):
    self.num_classes = num_classes
    self.train_dataloader = dataloader
    self.num_train_timesteps = num_train_timesteps
    self.net = ClassConditionedUNet(num_classes=num_classes, class_emb_size=class_emb_size, sample_size=sample_size, c_in=c_in, c_out=c_out).to(device)
    self.sample_size = sample_size
    self.channels = c_in
    self.device = device

    if scheduler=='ddim':
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps, beta_schedule=beta_schedule)
        self.noise_scheduler.set_timesteps(self.num_train_timesteps)
    else:
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps, beta_schedule=beta_schedule)
        
  
  def train(self, n_epochs=5, lr=1e-3):
    self.loss_fn = nn.MSELoss()
    self.opt = torch.optim.Adam(self.net.parameters(), lr)
    self.losses = []

    for epoch in range(n_epochs):
      for x,y in tqdm(self.train_dataloader):
        x = x.to(self.device)
        y = y.to(self.device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, self.num_train_timesteps-1, (x.shape[0],)).long().to(self.device)
        noisy_x = self.noise_scheduler.add_noise(x,noise,timesteps)

        pred = self.net(noisy_x, timesteps, y)
        loss = self.loss_fn(pred, noise)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.losses.append(loss.item())
  
  def sample(self):
    x = torch.randn(80, self.channels, self.sample_size, self.sample_size).to(self.device)
    y = torch.tensor([[i]*8 for i in range(self.num_classes)]).flatten().to(self.device)

    for i,t in tqdm(enumerate(self.noise_scheduler.timesteps)):
      with torch.no_grad():
        residual = self.net(x,t,y)

      x = self.noise_scheduler.step(residual, t, x).prev_sample
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')
    return x

if __name__ == "__main__":
  device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
  
  # import dataset
  mnist = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())

  # create dataloaders
  train_dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

  # generate images
  diffuser = Diffuser(dataloader=train_dataloader,
                    num_train_timesteps=1000, 
                    beta_schedule='squaredcos_cap_v2', 
                    num_classes=10, 
                    class_emb_size=4, 
                    sample_size=28, 
                    c_in=1, 
                    c_out=1)
  diffuser.train(n_epochs=150, lr=1e-4)
  samples = diffuser.sample()

  # generate 5040 images
  tests = []
  for _ in range(63):
    samples = diffuser.sample()
    tests.append(samples)
  
  # save images
  torch.save(tests, 'tests_ddpm.pt')

  # add class to images
  annotated_tests = []
  for batch in tests:
     for i in range(len(batch)):
        cur_num = int(i/8)
        annotated_tests.append((batch[i].cpu(), cur_num))
  
  # save images + class
  torch.save(annotated_tests, 'annotated_tests_ddpm.pt')
