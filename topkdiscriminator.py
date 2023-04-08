import torch
import torchvision 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models.mobilenet import mobilenet_v2
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class TopKDiscriminator:
    def __init__(self, model, device, dataloader, c_in_model, c_in_data, batch_size):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.c_in_model = c_in_model
        self.c_in_data = c_in_data
        self.batch_size = batch_size
    
    def fit(self, loss_fn=CrossEntropyLoss(), learning_rate=0.01, epochs=10):
        self.model.to(self.device)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in tqdm(range(epochs)):
            correct = 0
            for batch_idx, (data, label) in enumerate(self.dataloader):
                if self.c_in_model != self.c_in_data:
                    data = data.repeat(1,self.c_in_model,1,1)
                
                data, label = data.to(self.device), label.to(self.device)
                opt.zero_grad()
                y_hat = self.model(data)
                loss = loss_fn(y_hat, label)
                loss.backward()
                opt.step()
                pred = y_hat.argmax(dim=1, keepdim=True)
                results = label.eq(pred.view_as(label))
                correct += results.sum().item()

            acc = correct / len(self.dataloader.dataset)
            print("\n Accuracy this epoch = {}".format(acc))
    
    def predict(self, testloader, k=1):
        self.model.eval()
        correct = 0
        error_idx = []

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(testloader):
                if self.c_in_model != self.c_in_data:
                    data = data.repeat(1,self.c_in_model,1,1)

                data, label = data.to(self.device), label.to(self.device)
                y_hat = self.model(data)
                #pred = y_hat.argmax(dim=1, keepdim=True)
                results = torch.Tensor([True if x in y else False for x,y in zip(label, y_hat.topk(k=k).indices)]).type(torch.bool)
                incorrect_ids = [batch_idx*self.batch_size+id for id in range(len(results)) if results[id] == False]
                error_idx += incorrect_ids
                correct += results.sum().item()
        
        acc = correct / len(testloader.dataset)
        print("\n Accuracy = {}/{} , {}".format(correct,len(testloader.dataset),acc))

        return error_idx

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # data
    training_dataset = datasets.MNIST(root='mnist/', train=True,download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = datasets.MNIST(root='mnist/', train=False, transform=torchvision.transforms.ToTensor())
    full_dataset = training_dataset+test_dataset

    mnist_dataloader = DataLoader(full_dataset, batch_size=1000, shuffle=True)

    # load in model
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)

    # run discriminator
    discriminator = TopKDiscriminator(model, device, mnist_dataloader, 3, 1, 1000)
    discriminator.fit(CrossEntropyLoss(), 0.001, 20)

    # load in synthetic images
    ddpm_dataset = torch.load('./data/tests_ddpm_with_classes.pt')
    ddpm_dataloader = DataLoader(ddpm_dataset, batch_size=1000, shuffle=False)
    ddim_dataset = torch.load('./data/tests_ddim_with_classes.pt')
    ddim_dataloader = DataLoader(ddim_dataset, batch_size=1000, shuffle=False)

    # return errors
    ddpm_errors = discriminator.predict(ddpm_dataloader,k=1)
    ddim_errors = discriminator.predict(ddim_dataloader,k=1)
    ddpm_errors_k_2 = discriminator.predict(ddpm_dataloader,k=2)
    ddim_errors_k_2 = discriminator.predict(ddim_dataloader,k=2)

    # remove errors from synthetic dataset
    ddpm_corrected_k_1 = [val for idx,val in enumerate(ddpm_dataset) if idx not in ddpm_errors]
    ddim_corrected_k_1 = [val for idx,val in enumerate(ddim_dataset) if idx not in ddim_errors]
    ddpm_corrected_k_2 = [val for idx,val in enumerate(ddpm_dataset) if idx not in ddpm_errors_k_2]
    ddim_corrected_k_2 = [val for idx,val in enumerate(ddim_dataset) if idx not in ddim_errors_k_2]

    torch.save(ddpm_corrected_k_1, 'ddpm_corrected_k_1.pt')
    torch.save(ddim_corrected_k_1, 'ddim_corrected_k_1.pt')
    torch.save(ddpm_corrected_k_1, 'ddpm_corrected_k_2.pt')
    torch.save(ddim_corrected_k_1, 'ddim_corrected_k_2.pt')

