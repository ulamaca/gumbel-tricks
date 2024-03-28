'''
    demonstrate applying gumbel-softmax to choose many binary selection at once:
        we got N values,
        we want to chose for each of N from two (binary selection) s.t.
        their sum are maximal
'''

import torch
import numpy as np
from torch.nn.functional import gumbel_softmax

def create_random_array(N, seed=None):
    np.random.seed(seed)
    return np.random.rand(N, 2)

def create_torch_variables(N, seed=None):
    np.random.seed(seed)
    data = np.random.rand(N, 1)  # Create random array of size N-by-1
    data = np.concatenate([data, -data], axis=1)  # Concatenate with its negative for N-by-2
    tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)  # Convert to PyTorch tensor with requires_grad=True
    return tensor

if __name__ == '__main__':
    # Set random seed for reproducibility
    

    # 1 set numpy 
    device = 'cpu'
    tau = 5.0
    N = 100000
    seed = 42
    steps = 100
    learning_rate = 5.0

    ## TODO: one can study the effect of N & tau
    
    ## define variables in the formulation
    V = create_random_array(N, seed)
    gt = V.max(axis=1).sum()
    print(f"the max sum is: {gt}")

    V_tensor = torch.tensor(V, dtype=torch.float32)
    X = create_torch_variables(N, seed)

    X.to(device)
    V_tensor.to(device)

    optimizer = torch.optim.Adam([X], lr=learning_rate)

    for n in range(steps):
        y = torch.mul(gumbel_softmax(X, tau=tau), V_tensor) # V masked by gumbel of X
        loss = -y.sum(axis=1).sum() # the selcted values > sum of selected values (and want to minimize the neg > maximize)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step#{n+1}: sum={-loss.item()} (ground_truth={gt})")