# Adapted from https://github.com/davidmrau/mixture-of-experts/tree/master
# Author: Yuwei Sun


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import loralib as lora
import math
import torch.nn.functional as F
import copy

def print_trainable_status(model):
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

class LoRALinear(nn.Module):
    def __init__(self, lora_rank, linear_layer):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = lora_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.lora_a = nn.Parameter(torch.randn(lora_rank, linear_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, lora_rank))
        self.reset_parameters()
 
    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        torch.nn.init.zeros_(self.lora_b)

    def forward(self, input):
        # general implementation for lora (adding and scaling)
        adapter_out = torch.matmul(input, self.lora_a.T)
        adapter_out = torch.matmul(adapter_out, self.lora_b.T) / self.rank
        return F.linear(input, self.weight, self.bias) + adapter_out
    
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dim, dim, ff):
        super(MLP, self).__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.ff = copy.deepcopy(ff) # creat a new instance based on the target model
        self.ff.attention.query = LoRALinear(1,self.ff.attention.query)
        self.ff.attention.key = LoRALinear(1,self.ff.attention.key)
        self.ff.attention.value = LoRALinear(1,self.ff.attention.value)
        self.ff.output.dense = LoRALinear(1,self.ff.output.dense)

    def forward(self, hidden_states):

        hidden_states = hidden_states.reshape(-1,self.dim,self.hidden_dim)
        out = self.ff(hidden_states)[0]

        return out


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, hidden_dim, dim, num_experts, ff=None):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.input_size = input_size

        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, hidden_dim, dim, ff) for i in range(self.num_experts)])

 
    def forward(self, x, gate):
        # averge all the adapters outputs
        # out = torch.mean([self.experts[gate](x)*logits[gate] for gate in range(len(logits))])

        out = self.experts[gate](x)
        loss = None
    
        return out, loss

