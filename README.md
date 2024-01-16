# Coagulation and gelation phenomena
This repo contains the notebook we used for this project.

## project_coagulation_123.ipynb
This notebook contains the functions and main loop to compute the necessary results for part 1,2, and 3 of the project, which focus on a crude monte carlo algorithm to compute mean and confidence intervals for some key features in gelation phenomena, the concentration average, the moments, and the gelation time.

### Key components

### Functions

### Running the script


## plotting_123.ipynb

This notebook contains the loading and the plotting part for the results for part 2 and 3.


## Project Coagulation Importance
This part of the project focuses on the importance sampling of `c_k` in the context of coagulation processes. For simplicity this notebook contains the same functions as before, but with some key differences to achieve importance sampling

### Key Components

1. **Setting the seed**: For reproducibility, a seed is set at the beginning of the script using both the `random` and `numpy.random` modules.
2. **Defining parameters**: Several parameters are defined, including `N_val`, `t_val`, `importance_kernel`, `kernel_o`, `R`, and `k_vec`.
3. **Main loop**: The main part of the script is a loop over different values of `N` and `R`. For each combination of `N` and `R`, the script calculates the crude and importance sampling expectations and variances for `c_k` using the `crude_c` and `importance_sampling_c` functions. The results are stored in a list.
4. **Saving results**: The results are saved to a CSV file using `numpy.savetxt`.

### Functions

The script relies on two key functions:

- `crude_c(N, t, kernel_o, R, X0, k_vec)`: This function calculates the crude expectation and variance for `c_k`.
- `importance_sampling_c(N, t, importance_kernel, kernel_o, R, X0, k_vec)`: This function calculates the importance sampling expectation and variance for `c_k`.

### Running the Script

To run the script, simply open the `project_coagulation_importance.ipynb` file in a Jupyter notebook environment and execute the cells in order.
