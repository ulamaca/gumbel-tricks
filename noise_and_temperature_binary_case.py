'''
    to distinguish between the influence of noise and tempearture
        guess
            tau > control the sample fromat (more onehot or more uniform vec)
            noise > control certainty of samples
        
            20240407 Update: the distinction may not be needed since one can resort to ST (straight-through) trick to direct
                            optimize the only-one-choice sample by the clever (y_hard-y_soft).detach() + y_soft
                            I was not aware of it and had some misunderstanding. 
                            But the tau/noise trade-off might have some usecase in the future (?)

    plot
        scatter: x is the dim of the vector, y sampled values of the tensors
'''
import torch
import numpy as np
import pandas as pd
from torch.nn.functional import gumbel_softmax
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_samples = 15

    base_scale = 1.0
    noise_levels = [1e-3, 1e-2, 1e-2, 1.0, 10.0, 100.0]
    noise_levels = [base_scale*lv for lv in noise_levels]
    tau = 100 #1e-5 #100 #0.2

    seed_tensor = torch.tensor([5*base_scale, base_scale], dtype=torch.float32).to('cpu')    
    
    fig, axes = plt.subplots(ncols=len(noise_levels), nrows=1, figsize=(len(noise_levels) * 4, 8))

    for ax, noise in zip(axes, noise_levels):
        g_sources = []
        for _ in range(n_samples):
            if np.random.uniform(0.0,1.0) < 0.5:
                noise_kernel = torch.tensor([-noise, noise], dtype=torch.float32).to('cpu')
            else:
                noise_kernel = torch.tensor([noise, -noise], dtype=torch.float32).to('cpu')

            g_source = seed_tensor + noise * noise_kernel
            g_sources.append(g_source)
        
        g_samples = torch.stack(g_sources, axis=0)
        g_samples = gumbel_softmax(g_samples, tau, dim=1)
        g_samples = g_samples.detach().numpy()       
       
        # Scatter plot for 'choice-0'
        ax.scatter(np.zeros(g_samples.shape[0]), g_samples[:,0], alpha=0.5, label='Choice 0')
        # Scatter plot for 'choice-1', slightly offset on the x-axis for clarity
        ax.scatter(np.ones(g_samples.shape[0]) * 1, g_samples[:,1], alpha=0.5, label='Choice 1')

        # Setting labels and title
        ax.set_ylabel('sampled value')
        ax.set_title(f'noise level = {noise}, tau={tau}')
        plt.xticks([0, 1], ['Choice-0', 'Choice-1'])  # Set x-ticks to correspond to choices

        # Show legend
        ax.legend()        
    
    # Adjust layout
    plt.tight_layout()
    # Display the plot
    plt.show()
    breakpoint()
    
