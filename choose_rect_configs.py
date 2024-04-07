'''
    test how gumbel help choose configs of rectanges so that the area/ (width&heigh) becomes the minimal
'''

import numpy as np
import torch
from torch.nn.functional import gumbel_softmax, relu
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

def plot_results(X, W, names):
    '''
        by GPT-4, 3/31
    '''
    # Creating the plot again
    fig, ax = plt.subplots()

    # Adding rectangles to the plot again
    for (x, y), (width, height), name in zip(X, W, names):
        ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor='black', facecolor='yellow', alpha=0.5))
        ax.text(x, y, name, verticalalignment='bottom', horizontalalignment='left')

    # Correcting the calculation for the limits of the plot
    offset = 1
    ax.set_xlim(X[:,0].min()-offset, max([x + width for (x, _), (width, _) in zip(X, W)]) + offset)
    ax.set_ylim(X[:,1].min()-offset, max([y + height for (_, y), (_, height) in zip(X, W)]) + offset)

    plt.show()

## HYPERPARAMS
# use GPU when available when use_gpu tag is on, otherwise always use CPU (it is faster for my laptop :)
use_gpu = False
if use_gpu:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'

N = 10
Nc = 20 # num. of config candidates
tau = 5.0
steps = 2000
learning_rate = 1.0
which_wh = 0 # w or h to minimize (0=w, 1=h)
seed = 42
shape_control = 'area' # or 'wh'
anneal_rate = 0.99
min_tau = 0.05
##


## init
np.random.seed(seed)
config_W = np.random.randint(low=1, high=21, size=(Nc, 2))
X0 = 30.0*np.random.randn(N, 2) + 50.0

if __name__ == "__main__":
    '''
        _t for tensor
    '''
    torch.manual_seed(seed)   
    s_t = torch.randn(N, Nc).to(device=device).requires_grad_(True) # variable to optimize
    X = torch.from_numpy(X0).to(device=device).requires_grad_(True)

    optimizer = torch.optim.Adam([X, s_t], lr=learning_rate)
    
    print(f"using device = {device}")
    print("the optimization starts:")
    tau_i = tau # the init tau
    for i in range(steps):
        tau_i = max(tau_i * anneal_rate, min_tau)
        g_sel_t = gumbel_softmax(s_t, dim=1, tau=5.0, hard=True) # gumbel selector
        #g_sel_t_hard = gumbel_softmax(s_t, dim=1, hard=True)
        
        config_W_t = torch.from_numpy(config_W).float().to(device=device)

        W = torch.matmul(g_sel_t, config_W_t) 
        #print(W)       

        loss_o = torch_overlap_loss(X, W)
        loss_o = torch.tril(loss_o, diagonal=-1).sum()
        loss_w = torch_total_len_loss(X, W, which_dim=0)
        loss_h = torch_total_len_loss(X, W, which_dim=1)

        if shape_control == 'area':        
            loss = 100.0*loss_o + 5.0*loss_w*loss_h
        elif shape_control == 'wh':
            loss = 10.0*loss_o + 5.0*loss_w + 5.0*loss_h
        else:
            raise ValueError

        optimizer.zero_grad()  
        loss.backward()                
        optimizer.step() 

        print(f"step#{i}: total_w={loss_w.item()}, total_h={loss_h.item()}, overlap={loss_o.item()}")        

    # get the final results and plot it
    X_ = X.detach().cpu().numpy()
    g_sel_t_hard = gumbel_softmax(s_t, dim=1, hard=True)
    W_t = torch.matmul(g_sel_t_hard, config_W_t) 
    W_ = W_t.detach().cpu().numpy()
    
    names = [f'rect-{i+1}' for i in range(X_.shape[0])]

    plot_results(X_, W_, names)
    print("check the possible config_Ws and the selection results:")
    print(config_W, g_sel_t_hard.detach().cpu().numpy())
