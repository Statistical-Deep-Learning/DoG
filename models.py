import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, hidden_sizes=[512,256,128,64]):
        super().__init__()
        len_hiddens = len(hidden_sizes)
        self.encoder =  nn.ModuleList()
        for i in range(len_hiddens-2):
            self.encoder.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.encoder.append(nn.ReLU(0.2))
        
        self.mean = nn.Linear(hidden_sizes[-2], hidden_sizes[-1])
        self.log_var = nn.Linear(hidden_sizes[-2], hidden_sizes[-1])

        self.decoder =  nn.ModuleList()
        for i in range(len_hiddens-2):
            self.decoder.append(nn.Linear(hidden_sizes[len_hiddens-i-1], hidden_sizes[len_hiddens-i-2]))
            self.decoder.append(nn.ReLU(0.2))
        self.decoder.append(nn.Linear(hidden_sizes[1], hidden_sizes[0]),)
        self.decoder.append(nn.Sigmoid())
        
    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(log_var)
        z = mean + eps * torch.exp(log_var * 0.5)
        return z
    
    @torch.no_grad()
    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.mean(x)
        return x
    
    @torch.no_grad()
    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
    
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mean, log_var)
        for layer in self.decoder:
            z = layer(z)
        return z, mean, log_var  