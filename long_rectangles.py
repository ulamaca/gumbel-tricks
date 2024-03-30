'''
    test how gumbel help choose configs of rectanges so that the width becomes the maximal
'''

import numpy as np
import torch
from torch.nn.functional import gumbel_softmax, relu


def torch_overlap_loss(X, W, gamma=10.0):
    '''
        calculate the overlap of a set of rectanlces whose
            bottom left coordinates are defined by X
            size configs (width, height) are defined by W
    '''
    N = X.shape[0]
    assert X.shape[0] == W.shape[0], '# of rectangles are aligned with # of size configs'

    Wr = W[:,0].repeat(N,1)
    Hr = W[:,1].repeat(N,1)
    Xr = X[:,0].repeat(N,1)
    Yr = X[:,1].repeat(N,1)

    o_x = Xr+Wr-Xr.T
    o_x = torch.stack([
        o_x, o_x.T, Wr, Wr.T
    ], dim=0)
    o_x = torch.logsumexp(-o_x*gamma, dim=0)/ (-gamma)
    o_x = relu(o_x)

    o_y = Yr+Hr-Yr.T
    o_y = torch.stack([
        o_y, o_y.T, Hr, Hr.T
    ], dim=0)
    o_y = torch.logsumexp(-o_y*gamma, dim=0)/ (-gamma)
    o_y = relu(o_y)

    return o_x*o_y

def torch_total_len_loss(X, W, which_dim=0, gamma=10.0):
    '''
        soft version of max X+W - min X
    '''
    x = X[:, [which_dim]]
    ww = W[:, [which_dim]]    
    soft_total_len = torch.logsumexp((x+ww)*gamma, dim=0)/(gamma) - torch.logsumexp(x*-gamma, dim=0)/(-gamma)

    return soft_total_len

device = 'cpu'
N = 10
Nc = 20 # num. of config candidates
tau = 5.0
steps = 500
learning_rate = 5.0
which_dim = 0 # w or h to minimize (0=w, 1=h)

# TODO
# plot of the results

# init
config_W = np.random.randint(low=1, high=21, size=(Nc, 2))
X0 = 30.0*np.random.randn(N, 2) + 50.0

if __name__ == "__main__":
    '''
        _t for tensor
    '''
    s_t = torch.randn(N, Nc, requires_grad=True).to(device=device) # variable to optimize
    X = torch.from_numpy(X0).requires_grad_(True).to(device=device)

    optimizer = torch.optim.Adam([X, s_t], lr=learning_rate)
    
    for i in range(steps):
        g_sel_t = gumbel_softmax(s_t, dim=1, tau=5.0) # gumbel selector
        #g_sel_t_hard = gumbel_softmax(s_t, dim=1, hard=True)
        
        config_W_t = torch.from_numpy(config_W).float().to(device=device)

        W = torch.matmul(g_sel_t, config_W_t) 
        #print(W)       

        loss_o = torch_overlap_loss(X, W)
        loss_o = torch.tril(loss_o, diagonal=-1).sum()
        loss_l = torch_total_len_loss(X, W, which_dim=which_dim)

        loss = 10.0*loss_o + 5.0*loss_l

        optimizer.zero_grad()  
        loss.backward()                
        optimizer.step() 

        print(f"step#{i}: total_len={loss_l.item()}, ideal_len={config_W[:, which_dim].min()*N}")
        print(f"step#{i}: overlap={loss_o.item()}")

    print("selected WH")
    g_sel_t_hard = gumbel_softmax(s_t, dim=1, hard=True)
    print(torch.matmul(g_sel_t_hard, config_W_t) )
    breakpoint()
