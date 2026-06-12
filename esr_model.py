import torch
import torch.nn as nn

class ESRModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder=nn.Sequential(

            nn.Conv1d(1,32,5,padding=2),
            nn.ReLU(),

            nn.Conv1d(32,64,5,padding=2,stride=2),
            nn.ReLU(),

            nn.Conv1d(64,128,5,padding=2,stride=2),
            nn.ReLU()
        )

        self.decoder=nn.Sequential(

            nn.ConvTranspose1d(128,64,2,stride=2),
            nn.ReLU(),

            nn.ConvTranspose1d(64,32,2,stride=2),
            nn.ReLU(),

            nn.Conv1d(32,2,1)
        )

    def forward(self,x):

        x=x.unsqueeze(1)

        x=self.encoder(x)

        x=self.decoder(x)

        return x