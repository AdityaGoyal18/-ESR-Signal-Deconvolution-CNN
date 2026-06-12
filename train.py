import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from models.esr_model import ESRModel
from training.loss import esr_loss
from tqdm import tqdm

X=np.load("data/X.npy")
Y=np.load("data/Y.npy")

X=torch.tensor(X,dtype=torch.float32)
Y=torch.tensor(Y,dtype=torch.float32)

dataset=TensorDataset(X,Y)

loader=DataLoader(dataset,batch_size=64,shuffle=True)

model=ESRModel()

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=80

for epoch in range(epochs):

    total=0

    for x,y in tqdm(loader):

        optimizer.zero_grad()

        pred=model(x)

        loss=esr_loss(pred,y,x)

        loss.backward()

        optimizer.step()

        total+=loss.item()

    print("Epoch",epoch,total/len(loader))

torch.save(model.state_dict(),"esr_model.pth")