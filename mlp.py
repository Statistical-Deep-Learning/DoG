import torch.nn as nn
import torch
import math



def timestep_embedding(timesteps, dim=1, max_period=10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1000,1000,1000,1000,1000], cond_sizes=64, t_sizes=64, mod_ch=64, cdim=10, res=0):
        super().__init__()
        # Initialize layers
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.cond_sizes = cond_sizes
        self.t_sizes = t_sizes
        self.res = res
        self.mod_ch = mod_ch
        self.cdim = cdim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], input_size))

        self.act = nn.ReLU()

        self.temb_layer = nn.Sequential(
            nn.Linear(self.mod_ch, t_sizes),      
            nn.ReLU(),
            nn.Linear(t_sizes, input_size),
        )
        self.cemb_layer = nn.Sequential(
            nn.Linear(self.cdim, cond_sizes),   
            nn.ReLU(),
            nn.Linear(cond_sizes, input_size),
        )

    
    def forward(self, x, t, cemb=None):
        #get t and condition embedding
        t_emb, cond_emb = self.get_embeddings(t, cemb)
        #merge input, t emb and condition emb
        x = x.reshape(x.shape[0], self.input_size)     
        x = x + t_emb + cond_emb

        for layer in self.layers[:-1]:
            h = x                         #for residual
            x = torch.relu(layer(x))
            if self.res:
                x += h
        x = self.layers[-1](x)
        return x

    
    def get_embeddings(self, t, cemb):
        t_emb = self.temb_layer(timestep_embedding(t, self.mod_ch))
        c_emb = self.cemb_layer(cemb)

        return t_emb, c_emb