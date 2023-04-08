import torch
from torchvision import datasets
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
from LeNet5 import LeNet5

def train_model(device, model, train_loader, test_loader, loss_fn=CrossEntropyLoss(), epochs=100, learning_rate=0.01):
    model.to(device)
    sgd = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            sgd.zero_grad()
            pred = model(data.float())
            loss = loss_fn(pred, label.long())
            loss.backward()
            sgd.step()
        
        all_correct_num = 0
        all_sample_num = 0
        model.eval()

        for idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data.float()).detach()
            pred = torch.argmax(pred, dim=-1)
            current_correct_num = pred == label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        
        acc = all_correct_num / all_sample_num
        print("accuracy: ", acc)
    
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='data', train=False, transform=ToTensor())
    batch_size = 256
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    # load in model
    model = LeNet5().to(device)

    # train model
    model = train_model(device, model, train_loader, test_loader)

    # save model
    torch.save(model.state_dict(), 'lenet5.pt')



