'''
    test how gumbel help choose configs of rectanges so that the area/ (width&heigh) becomes the minimal
    + 1 visualize learning dynamics of Gumbel variables
        > observation-1: the selector converges at the beginning (fast converge to suboptimal!?)
        > observation-2: the tau won't influence the fast convergence property!
            >> TODO: to look at the effect of sampling, use multiple sample at each step for computing loss and do gradient descent
        > observation-3: if turn on normalize_logits, it will produce weird dynamics, not making sense! (TODO: to understand that)
        > TODO: may visualize with plots
'''

import numpy as np
import torch
from torch.nn.functional import gumbel_softmax, relu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.distributions import Categorical


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
Nc = 50 # num. of config candidates
tau = 1e6
steps = 2000
learning_rate = 1.0
which_wh = 0 # w or h to minimize (0=w, 1=h)
seed = 42
shape_control = 'area' # or 'wh'
anneal_rate = 0.99
min_tau = 0.05
normalize_logits = False
show_dynamics = True
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
    
    decision_trace = {}
    entropy_sel_trace = {}
    loss_trace = {}

    for i in range(steps):
        tau_i = max(tau_i * anneal_rate, min_tau)
        if normalize_logits:
            s_t = torch.nn.Softmax(dim=1)(s_t)            
            
        g_sel_t = gumbel_softmax(s_t, dim=1, tau=5.0, hard=True) # gumbel selector
        
        if i% 100 ==0:
            tmp = s_t.detach()
            decision_trace[i] = tmp.argmax(dim=1).numpy()
            p_t = tmp.exp()/tmp.exp().sum(dim=1).reshape(-1,1)      

            # two ways to observe entropy      
            #entropy_sel_trace[i] = [Categorical(p_t[i,:]).entropy().item() for i in range(p_t.shape[0])]            
            entropy_sel_trace[i] = p_t.max(dim=1)[0] - p_t.min(dim=1)[0]
        
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
        loss.backward(retain_graph=True)                
        optimizer.step() 

        loss_trace[i] = loss_w.item() * loss_h.item()
        print(f"step#{i}: total_w={loss_w.item()}, total_h={loss_h.item()}, overlap={loss_o.item()}")        

    # get the final results and plot it
    X_ = X.detach().cpu().numpy()
    g_sel_t_hard = gumbel_softmax(s_t, dim=1, hard=True)
    W_t = torch.matmul(g_sel_t_hard, config_W_t) 
    W_ = W_t.detach().cpu().numpy()
    
    names = [f'rect-{i+1}' for i in range(X_.shape[0])]

    breakpoint()
    plot_results(X_, W_, names)
    print("check the possible config_Ws and the selection results:")
    print(config_W, g_sel_t_hard.detach().cpu().numpy())

    if show_dynamics:
        print('dynamics:')
        for step, entropy in entropy_sel_trace.items():
            print(f"entropy@step={step}: ", entropy)
        
        for step, dec in decision_trace.items():
            print(f"decision@step={step}: ", dec)
