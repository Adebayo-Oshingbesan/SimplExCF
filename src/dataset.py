import numpy as np
import pandas as pd
import torch
import torchvision

import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

class TabularDataset(Dataset):
    def __init__(self, train, target_class, data = 'adult'):        
        datatypes = ['adult, diabetes','hospital', 'insurance']
        
        if data == 'adult':
            df = pd.read_csv('data/adult.csv')
            target = 'income'
            df[target] = df[target].map({'<=50K' : 0, '>50K' : 1})
        elif data == 'diabetes':
            cols = ['Diabetes', 'GenHlth', 'Age', 'Education', 'Income', \
                    'HighBP','BMI','HighChol','DiffWalk',\
                    'HeartDiseaseorAttack','PhysHlth',\
                    'HvyAlcoholConsump','Sex','CholCheck']
            df = pd.read_csv('data/diabetes_small.csv')[cols]
            target = 'Diabetes'
            df[target] = (df[target] > 0).astype(int)
        elif data == 'hospital':
            df = pd.read_csv('data/hospital_appointments.csv')
            target = 'No-show'
            df[target] = df[target].map({'No' : 0, 'Yes' : 1})
        elif data == 'insurance':
            df = pd.read_csv('data/insurance.csv')[['smoker', 'tier', 'region', 'sex', 'charges', 'bmi', 'children', 'age']]
            target = 'charges'
        else:
            assert False, f'data should be one of {datatypes}'
        
        self.columns = list(df.drop(columns = target).columns)
        self.encoder = ce.OrdinalEncoder()
        self.scaler = StandardScaler()
        
        if data != 'insurance':
            Y = df[target].values
        else:
            Y = df[target].values/100_000
        data = self.encoder.fit_transform(df.drop(columns = target), Y)
        data = self.scaler.fit_transform(data).astype(np.float32)
        
         
        i = int(0.9 * len(data))
        if train is True:
            self.X = data[:i]
            self.y = Y[:i]
        else:
            self.X = data[i:]
            self.y = Y[i:]
            
        if target_class is not None:
            bool_idx = self.y == target_class
            self.X = self.X[bool_idx]
            self.y = self.y[bool_idx]
            
        self.shape = self.X.shape

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return [self.X[idx], self.y[idx]]
    
    
def load_adult(
    batch_size: int, train: bool, shuffle: bool = True, target_class = None) -> DataLoader:
    dataset = TabularDataset(train, target_class = target_class, data = 'adult')
    encoder, scaler, shape = dataset.encoder, dataset.scaler, dataset.shape
    if train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), encoder, scaler, shape
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
def load_diabetes(
    batch_size: int, train: bool, shuffle: bool = True, target_class = None) -> DataLoader:
    dataset = TabularDataset(train, target_class = target_class, data = 'diabetes')
    encoder, scaler, shape = dataset.encoder, dataset.scaler, dataset.shape
    if train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), encoder, scaler, shape
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
def load_hospital(
    batch_size: int, train: bool, shuffle: bool = True, target_class = None) -> DataLoader:
    dataset = TabularDataset(train, target_class = target_class, data = 'hospital')
    encoder, scaler, shape = dataset.encoder, dataset.scaler, dataset.shape
    if train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), encoder, scaler, shape
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
def load_insurance(
    batch_size: int, train: bool, shuffle: bool = True, target_class = None) -> DataLoader:
    dataset = TabularDataset(train, target_class = target_class, data = 'insurance')
    encoder, scaler, shape = dataset.encoder, dataset.scaler, dataset.shape
    if train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), encoder, scaler, shape
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
def load_mnist(
    batch_size: int, train: bool, shuffle: bool = True
) -> DataLoader:
    dataset = torchvision.datasets.MNIST(
        "./data/",
        train=train,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def data_transforms(phase = None):
    
    if phase == 'train':

        data_T = torchvision.transforms.Compose([
            
                torchvision.transforms.Resize(size = (256,256)),
                torchvision.transforms.RandomRotation(degrees = (-20,+20)),
                torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    elif phase == 'test':

        data_T = torchvision.transforms.Compose([

                torchvision.transforms.Resize(size = (224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    return data_T

def load_xray(
    batch_size: int, train: bool, shuffle: bool = True
) -> DataLoader:
    if train:
        dataset = torchvision.datasets.ImageFolder('data/chest_xray/train', transform=data_transforms('train'))
    else:
        dataset = torchvision.datasets.ImageFolder('data/chest_xray/test', transform=data_transforms('test'))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_m_mnist(
    batch_size: int, train: bool, shuffle: bool = True
) -> DataLoader:
    if train: 
        dataset = torchvision.datasets.ImageFolder('data/M_MNIST/Train', transform=data_transforms('train'))
    else:
        dataset = torchvision.datasets.ImageFolder('data/M_MNIST/Test', transform=data_transforms('test'))
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)