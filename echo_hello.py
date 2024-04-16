'''
    Apply Gumble Straight-Through training a model, able to learn to say hello!
    +1 v1: just learn to say something from characters of H,E,L,O
    +2 v2: can say HELLO exactly
    +3 v3: added option for using log_gumbel_softmax
        > TODO: think about it why? log_softmax, if set eps smaller, it will make the gradient value larger 
    +4 added noise-adding support: TODO, I want to make the learning explore properly, how to get this done as I want 

    0415, observing the gradients during leraning at the pre-gumbel logit layer    
    0416, observations: 
        (1) no matter hard/soft gumbel, the grad patterns are similar; (2) tau affects the scale of grad but not the pattern
        it might be due to the nature of the loss, one direction is to go for a more explorative problem


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import torch.optim as optim

class CharMLP(nn.Module):
    '''
        model are designed to be small
    '''
    def __init__(self, latent_dim=50, output_chars=5, char_set_size=26):
        super(CharMLP, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_chars * char_set_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Reshape to (batch_size, output_chars, char_set_size)
        return x.view(-1, 5, 26)
    

## vocab handling
# dict to map: 
    # 0:A, 1:B ... 25:Z
alphabet_vocab = {i: chr(97 + i).upper() for i in range(26)}
char_to_index = {v:k for k,v in alphabet_vocab.items()}

def indices_to_sequences(indices, mapping_dict=alphabet_vocab):
    batch_size, seq_length = indices.shape
    # Initialize an empty list to store character sequences
    character_sequences = []

    # Iterate over each sequence in the batch
    for i in range(batch_size):
        # Convert each index in the sequence to its corresponding character
        sequence = ''.join([mapping_dict[idx.item()] for idx in indices[i]])
        # Append the formed sequence to the list
        character_sequences.append(sequence)

    return character_sequences

def get_HELLO_idxs():
    return [i for i,v in alphabet_vocab.items() if v in ['H', 'E', 'L', 'O']]
##

## losses
def count_a(output, a_index=get_HELLO_idxs()):
    # Assuming 'a' is the first character in the character set
    return output[:, :, a_index].sum()

def count_a_per_datapoint(output, a_index=0):
    # Assuming 'a' is the first character in the character set
    # This will return a tensor of shape [batch_size], with each entry being the count of 'A's in that data point
    return output[:, :, a_index].sum(dim=1)

def count_hello(output):    
    losses = []
    for i, char in enumerate(['H', 'E', 'L', 'L', 'O']):
        loss = output[:, i, char_to_index[char]].sum()    
        losses.append(loss)
    
    return sum(losses)

## hooks
def print_logit_grad_hook(module, grad_input, grad_output):
    print("the mean grad value over a mini-batch")
    print(grad_output[0].mean(dim=0))    
    print("---")

if __name__ == "__main__":
    # hperparams
    latent_dim = 16
    anneal_rate = 0.95
    lr = 0.001
    steps = 10
    batch_size = 5
    tau = 10000.0
    n_inference = 50         
    loss_types = ['a', 'helo-char', 'hello']
    loss_type = loss_types[2]
    show_grad = True
    add_noise = False
    noise_level = 100.0
    log_softmax = False
    log_softmax_eps = 1e-1

    # model setup
    model = CharMLP(latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if show_grad:
        model.register_backward_hook(print_logit_grad_hook)


    # training loop
    tau_step = tau
    for step in range(steps):
    
        latent_var = torch.randn(batch_size, latent_dim)
        # Get the logits for each character
        logits = model(latent_var)
        
        if step % 10 == 0 and step > 0:
            tau_step = tau_step* anneal_rate
        
        if add_noise:
            # noise will be at the 'noise_level' of logits' exact value
            noise = logits.max(dim=2, keepdim=True)[0] * noise_level * torch.randn_like(logits)
            logits = logits + noise
            
        # Convert logits to probabilities        
        one_hot_samples = gumbel_softmax(logits, tau=tau_step, hard=True, dim=-1) # over the dimension of characters
        if log_softmax:
            one_hot_samples = torch.log(one_hot_samples+log_softmax_eps)
        
        


        # optimize
        if loss_type == 'a':
            loss = -count_a(one_hot_samples, ['A'])
        elif loss_type == 'hello':
            loss = -count_hello(one_hot_samples)
        elif loss_type == 'helo-char':
            loss = -count_a(one_hot_samples)
        else:
            raise ValueError
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step={step}, tau={tau_step}: loss={loss.item()}")
        
    # demo inference:       
    latent_var_inference = torch.randn(n_inference, latent_dim)
    logits = model(latent_var_inference)
    one_hot_samples = gumbel_softmax(logits, tau=tau_step, hard=True, dim=-1)
    _, index_samples = torch.max(one_hot_samples, dim=2)    
    char_samples = indices_to_sequences(index_samples)
    breakpoint()
    
    

