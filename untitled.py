import numpy as np 
import random
import matplotlib.pyplot as plt
import scipy.stats as st 
from scipy.stats import norm
from scipy.stats import burr12
from scipy.fft import fft, ifft
import scipy.linalg as linalg
from scipy.optimize import minimize
import collections 
import time
from itertools import chain
import timeit
from coagulation_functions import simulate_coagulation_optimized, MC_coagulation_optimized, mc_averaging


def main():
    N_values = [100]
    k_values = [1, 5, 15, 50]
    p = 1.5
    kernels = ['kernel_k3', 'kernel_k4']
    alpha = 0.05  # Confidence level for averaging

    # Loop over each value of N
    for N in N_values:
        # Loop over each kernel
        for kernel in kernels:
            # Create a figure with subplots for each k value
            fig, axs = plt.subplots(1, len(k_values), figsize=(20, 5))
            fig.suptitle(f'Concentration plots for N = {N}, {kernel}')

            # Iterate over each value of k
            for k_idx, k in enumerate(k_values):
                X0 = np.ones(N)  # Initial condition

                # Run MC Coagulation Optimized
                Zi_conc, Zi_mom, Zi_gel, Ji = MC_coagulation_optimized(N, T=10, R=10, k=k, X0=X0, p=p, kernel=kernel)

                # Perform MC Averaging
                means, sigma, J_sorted, lower_bound, upper_bound = mc_averaging(Ji, Zi_conc, alpha, N)

                # Plot the results in the respective subplot
                axs[k_idx].plot(J_sorted, means, label=f'k = {k}', linewidth=0.9, color="black")
                axs[k_idx].fill_between(J_sorted, lower_bound, upper_bound, color="red", alpha=0.3)
                axs[k_idx].set_title(f'k = {k}')
                axs[k_idx].set_xlabel('Time')
                x_max = np.ceil(max(J_sorted))
                axs[k_idx].set_xticks(np.arange(0, x_max + 1, 1))

                if k_idx == 0:
                    axs[k_idx].set_ylabel('Average concentration')

                axs[k_idx].legend()

            # Adjust layout and show plot
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
