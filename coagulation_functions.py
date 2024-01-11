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
from multiprocessing import Pool

def kernel_k1(x, y):
    return 1 if x > 0 and y > 0 else 0

def kernel_k2(x, y):
    return x * y

def kernel_k3(x, y):
    return (x + y) if x > 0 and y > 0 else 0

def kernel_k4(x, y):
    return 1/4*(x**(1/3) + y**(1/3))**3 if x > 0 and y > 0 else 0

def kernel_ka1(x,y):
    return (x*y)**0.7

def kernel_ka2(x,y):
    return (x*y)**0.8

def kernel_ka3(x,y):
    return (x*y)**0.9
    
kernel_functions = {
    'kernel_k1': kernel_k1,
    'kernel_k2': kernel_k2,
    'kernel_k3': kernel_k3,
    'kernel_k4': kernel_k4,
    'kernel_ka1': kernel_ka1,
    'kernel_ka2': kernel_ka2,
    'kernel_ka3': kernel_ka3,
}



#function to compute K(x,y) the entries of our kernels and the denominator to use in the index distribution
def kernel_sum(particles, kernel):
    start = timeit.default_timer()
    n = len(particles)
    weights = np.zeros((n, n))
    
    kernel = kernel_functions[kernel]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i, j] = kernel(particles[i], particles[j])
    
    total_weight = np.sum(weights)
    stop = timeit.default_timer()
    #print("normal",stop - start)
    return weights, total_weight


#function to compute K(x,y) the entries of our kernels and the denominator to use in the index distribution
def kernel_sum_triangular(particles, kernel):
    start = timeit.default_timer()
    n = len(particles)
    weights = np.zeros((n, n))
    
    kernel = kernel_functions[kernel]
    
    for i in range(n):
        for j in range(i+1, n):
                weights[i, j] = kernel(particles[i], particles[j])
    
    weights += np.triu(weights, 1).T  
    
    total_weight = np.sum(weights)
    stop = timeit.default_timer()
    #print("Tri",stop - start)
    return weights, total_weight


#function that calculates the index distribution matrix
def calculate_P_index(particles, kernel):
    #weights, total_weight = kernel_sum(particles, kernel)
    weights, total_weight = kernel_sum_triangular(particles, kernel)
    
    P_index = weights / total_weight

    return P_index

# function to calculate the majorant kernel as the maximum value of the collision kernel
def majorant_kernel(particles,kernel):
    kernel_func = kernel_functions[kernel]
    weights, total_weight = kernel_sum_triangular(particles, kernel)
    return np.argmax(weights)


# Acceptance-Rejection algorithm sampling function, as majorant kernel we choose the maximum value of the kernel
def AR_sampling(particles, kernel, recursion_counter=0):
    start = timeit.default_timer()

    kernel_func = kernel_functions[kernel]
    Max = majorant_kernel(particles, kernel)

    U = random.uniform(0, 1)
    row_index = random.randint(0, len(particles) - 1)
    col_index = random.randint(0, len(particles) - 1)
    while col_index == row_index:
        col_index = random.randint(0, len(particles) - 1)

    if U <= kernel_func(row_index, col_index) / Max:
        i, j = row_index, col_index

        stop = timeit.default_timer()
        print("AR time:", stop - start, "iterations:", recursion_counter)

        return i, j

    else:
        recursion_counter += 1
        return AR_sampling(particles, kernel, recursion_counter)

    
#function to sample from the index distribution in a simple way: 
def inverse_sampling(particles, kernel):
    
    start = timeit.default_timer()
    
    U = random.uniform(0, 1)
    
    P_idx = calculate_P_index(particles, kernel)
    
    #find first index s.t. it is bigger than U
    flat_index = np.where(np.cumsum(P_idx) >= U)[0][0]
    
    #get the 2D index from the flat one
    i, j = np.unravel_index(flat_index, P_idx.shape)
    
    stop = timeit.default_timer()
    #print("inverse time", stop-start)
        
    return i,j


#function to compute the transition matrix for index i,j for the Markov process described in (3) from the project description
def transition_matrix(i,j, N):
    M = np.eye(N)
    M[i][j]=1
    M[j][j]=0
    
    return M
    
#to compute parameters to average

def average_conc(X,k, N):
    counts = []
    
    for arr in X:
        count = np.count_nonzero(arr == k)
        counts.append(count/N)
        
    return counts

def moments(X,p):
    X = np.array(X)
    mom_p = np.mean(X**p, axis= 1)
    return mom_p


def gelation(X, N):
    X = np.array(X)
    G =  np.max(X, axis=1)/N
    return G


def simulate_coagulation_optimized(N, T, X0, kernel):
    kernel_func = kernel_functions[kernel]
    current_time = 0
    jump_count = 0

    Xt = [X0.copy()]
    t = [current_time]

    while current_time < T and jump_count <= N:
        weights, total_weight = kernel_sum_triangular(X0, kernel)
        Lambda = total_weight / (2 * N)
        Sn = st.expon.rvs(scale=1 / Lambda)

        current_time += Sn
        jump_count += 1

        i, j = inverse_sampling(X0, kernel=kernel)
        M = transition_matrix(i, j, N)
        X0 = M.dot(X0)

        Xt.append(X0.copy())
        t.append(current_time)

    return Xt, t


def single_simulation(N, T, X0, kernel, k, p, r):
    Xt, t = simulate_coagulation_optimized(N=N, T=T, X0=X0, kernel=kernel)
    z_conc = average_conc(Xt, k, N)
    z_mom = moments(Xt, p)
    z_gel = gelation(Xt, N)
    return z_conc, z_mom, z_gel, t


def MC_coagulation_optimized(N, T, R, k, X0, p, kernel):
    # Preallocate numpy arrays for efficiency
    Zi_conc = np.zeros((R, N-1)) 
    Zi_mom = np.zeros((R, N-1))
    Zi_gel = np.zeros((R, N-1))
    Ji = np.zeros((R, N-1))

    # Prepare arguments for parallel processing
    args = [(N, T, X0, kernel, k, p, r) for r in range(R)]

    # Parallel processing
    with Pool() as pool:
        results = pool.starmap(single_simulation, args)

    for i, (z_conc, z_mom, z_gel, t) in enumerate(results):
        Zi_conc[i] = z_conc
        Zi_mom[i] = z_mom
        Zi_gel[i] = z_gel
        Ji[i] = t

    return Zi_conc, Zi_mom, Zi_gel, Ji



def mc_averaging(Ji, Zi, alpha, N):
    J_no_first = [sublist[1:] for sublist in Ji]
    J_no_first[0].append(0)
    J_sorted = sorted(list(chain.from_iterable(J_no_first)))
    J_sorted = np.array(J_sorted)

    Ji = np.array(Ji)

    means = []
    sigma = []

    for k in range(len(J_sorted)):
        condition = Ji <= J_sorted[k]
        Ji_smaller = np.where(condition, Ji, np.nan)
        max_indices = np.nanargmax(Ji_smaller, axis=1)

        Z_mean = [Zi[j][max_indices[j]] for j in range(len(max_indices))]

        means.append(np.mean(Z_mean))
        sigma.append(np.std(Z_mean, ddof=1))

    means = np.array(means)
    sigma = np.array(sigma)

    c_ok = st.norm.ppf(1-alpha/2)
    lower_bound = means - c_ok * sigma / np.sqrt(N)
    upper_bound = means + c_ok * sigma / np.sqrt(N)

    return means, sigma, J_sorted, lower_bound, upper_bound

