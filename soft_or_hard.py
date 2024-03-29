'''
    Will the gumbel param tend to be a soft decision or a hard one?

    Problem
        a list of values, with repreated highest, see how will the gumbel choose the min
    
    Observations:
        1. large tau will make the decision shifting 
            >> higher tau can make it more explorative, but still can converge
            >> annealing schedule is also reasonable
        
        2. this example, can be extended with more values and study
            >> 2.1 fluctuation rate of decisions at different tau's
            >> 2.2 fluctuation rate of decisions at different # of values
            >> 2.3                               at different # of optimals
            

'''

import torch
import numpy as np
from torch.nn.functional import gumbel_softmax

values = [3.5, 8.0, 9.0, 7.2, -1.0, -3.2, 2.5, 20.3, -3.2, 4.5, -3.2]


if __name__ == '__main__':
    # Set random seed for reproducibility
    

    # 1 set numpy 
    device = 'cpu'
    tau = 100.0
    seed = 42
    steps = 1000
    learning_rate = 5.0

    v_tensor = torch.tensor([values], dtype=torch.float32).to(device)
    x = torch.randn((1,len(values)), dtype=torch.float32, requires_grad=True).to(device)

    optimizer = torch.optim.Adam([x], lr=learning_rate)

    for n in range(steps):
        mul =  torch.mul(gumbel_softmax(x, tau=tau), v_tensor)
        loss = mul.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step#={n}: loss={loss.item()}, groun-truth={min(values)}")
        print(f"\t gumbel-value (fixed tau)", gumbel_softmax(x, tau=tau))
        print(f"\t gumbel-value (hard):", gumbel_softmax(x, hard=True))


    