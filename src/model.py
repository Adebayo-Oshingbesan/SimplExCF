import os
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from src.dataset import load_adult, load_diabetes, load_hospital, load_insurance, load_mnist, load_m_mnist, load_xray

class TabularModel(torch.nn.Module):
    def __init__(self, dim, out, n_cat: int, latent_dim = 300) -> None:
        super().__init__()
        self.n_cat = n_cat
        self.latent_dim = latent_dim
        
        self.lin1 = nn.Linear(dim, 1000)
        self.lin2 = nn.Linear(1000, latent_dim)
        self.lin3 = nn.Linear(latent_dim, out)
        self.bn1 = nn.BatchNorm1d(dim-n_cat)
        self.drops = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_disc, x_cont = x[:, : self.n_cat], x[:, self.n_cat :]
        x_cont = self.bn1(x_cont)
        x = torch.cat([x_cont, x_disc], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        return x

class MnistModel(torch.nn.Module):
    def __init__(self, dim, out):
        super(MnistModel, self).__init__()
        self.latent_dim = dim
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, out)
        
    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_representation(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
    
class XrayModel(torch.nn.Module):
    def __init__(self,dim,out):
        super(XrayModel,self).__init__()  
        self.latent_dim = dim
        
        self.conv1=nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()        
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        self.fc=nn.Linear(in_features=32 * 112 * 112,out_features=self.latent_dim)       
        self.fc1=nn.Linear(in_features=self.latent_dim,out_features=out)
        
    def latent_representation(self,inp):
        output=self.conv1(inp)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool(output)
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)            
        output=output.view(-1,32*112*112)
        output=self.relu4(output)
        output=self.fc(output)
        return output
        
    def forward(self,inp):
        output=self.latent_representation(inp)
        output=self.relu4(output)
        output=self.fc1(output)      
        return F.log_softmax(output, dim=-1)
    

def train_model(
    device: torch.device,
    dataset = 'adult',
    n_epoch: int = 1,
    batch_size_train: int = 64,
    batch_size_test: int = 1000,
    random_seed: int = 42,
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    log_interval: int = 100,
    model_reg_factor: float = 0.01,
    cv: int = 0,
):
    torch.random.manual_seed(random_seed + cv)
    torch.backends.cudnn.enabled = False
    
    out_dim = 2
    if dataset in ['adult','diabetes','hospital', 'insurance']:
        if dataset == 'adult':
            train_loader, encoder, scaler, shape = load_adult(batch_size_train, train=True)
            test_loader = load_adult(batch_size_test, train=False)
        elif dataset == 'diabetes':
            train_loader, encoder, scaler, shape = load_diabetes(batch_size_train, train=True)
            test_loader = load_diabetes(batch_size_test, train=False)
        elif dataset == 'hospital':
            train_loader, encoder, scaler, shape = load_hospital(batch_size_train, train=True)
            test_loader = load_hospital(batch_size_test, train=False)
        elif dataset == 'insurance':
            train_loader, encoder, scaler, shape = load_insurance(batch_size_train, train=True)
            test_loader = load_insurance(batch_size_test, train=False)
        
        if dataset != 'insurance':
            model = TabularModel(shape[1], 2, len(encoder.cols))
        else:
            model = TabularModel(shape[1], 1, len(encoder.cols), task = 'regression')
        task = 'tabular'
    elif dataset == 'mnist':
        train_loader = load_mnist(batch_size_train, train=True)
        test_loader = load_mnist(batch_size_test, train=False)
        out_dim = 10
        model = MnistModel(300,out_dim)
        task = 'non-tabular'
    elif dataset in ['pneumonia', 'm_mnist']:
        if dataset == 'pneumonia':
            train_loader = load_xray(batch_size_train, train=True)
            test_loader = load_xray(batch_size_test, train=False)
            model = XrayModel(300,out_dim)
            task = 'non-tabular'
        elif dataset == 'm_mnist':
            train_loader = load_m_mnist(batch_size_train, train=True)
            test_loader = load_m_mnist(batch_size_test, train=False)
            out_dim = 6
            model = XrayModel(300,out_dim)
            task = 'non-tabular'

    # Create the model
    model.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=model_reg_factor,
    )

    # Train the model
    train_losses = []
    train_counter = []
    test_losses = []

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            #data = data.reshape(-1, 28 * 28)
            data = data.to(device).float()
            target = target.to(device).float()
            optimizer.zero_grad()
            output = model(data)
            if dataset == 'insurance':
                loss = F.l1_loss(output, target)
            else:
                loss = F.nll_loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                #data = data.reshape(-1, 28 * 28)
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                if dataset == 'insurance':
                    test_loss += F.l1_loss(output, target, reduction="sum").item()
                else:
                    test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
            f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
        )

    test()
    for epoch in range(1, n_epoch + 1):
        train(epoch)
        test()
    if  task == 'tabular':
        return model, encoder, scaler
    else:
        return model
