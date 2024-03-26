'''
    demonstrate applying gumbel-softmax to solve a simple choice-then-sum task
'''


import torch
from torch.nn.functional import gumbel_softmax


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)    

    device = 'cpu'
    tau = 5
    x1 = torch.randn(1, 2, requires_grad=True).to(device=device)
    x2 = torch.randn(1, 2, requires_grad=True).to(device=device)

    v1 = torch.tensor([[100.0, 0.2]], dtype=torch.float32).to(device=device)
    v2 = torch.tensor([[-25, 30]], dtype=torch.float32).to(device=device)

    # optimization
    steps = 1000
    learning_rate = 0.1
    optimizer = torch.optim.Adam([x1, x2], lr=learning_rate)

    for step in range(steps):
        x1_ = gumbel_softmax(x1, tau=tau)
        x2_ = gumbel_softmax(x2, tau=tau)

        loss = -(torch.mul(x1_, v1) + torch.mul(x2_, v2)).sum()
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
    
        # Update parameters
        optimizer.step() 

        print(f"step#{step}: sum={-loss.item()}")

    
    
    breakpoint()