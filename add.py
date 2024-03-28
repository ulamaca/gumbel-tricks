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

    v1 = torch.tensor([[32.0, 0.2]], dtype=torch.float32).to(device=device)
    v2 = torch.tensor([[-25, 30]], dtype=torch.float32).to(device=device)

    # optimization
    steps = 200
    learning_rate = 1.0
    optimizer = torch.optim.Adam([x1, x2], lr=learning_rate)
    n_sample = 100

    for step in range(steps):
        losses = []
        for sample in range(n_sample):
            x1_ = gumbel_softmax(x1, tau=tau)
            x2_ = gumbel_softmax(x2, tau=tau)

            loss_s = -(torch.mul(x1_, v1) + torch.mul(x2_, v2)).sum()
            losses.append(loss_s)
        
        loss = sum(losses)/n_sample
        optimizer.zero_grad()  
        loss.backward()                
        optimizer.step() 

        print(f"step#{step+1}: sum={-loss.item()}")

    
    # inference
    x1_ = gumbel_softmax(x1, hard=True)
    x2_ = gumbel_softmax(x2, hard=True)
    sum_ = -(torch.mul(x1_, v1) + torch.mul(x2_, v2)).sum().item()
    print(f"the inferred result={-sum_}")
    print(f'\t with x1={x1_}, x2={x2_}')
    
    breakpoint()