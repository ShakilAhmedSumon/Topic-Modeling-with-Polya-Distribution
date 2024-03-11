# Estimation, Detection and Filtering
# Project: Topic modeling with Polya Distribution
# Authors: Shakil Shamed Sumon, Derek Helmes


import numpy as np
from scipy.stats import betabinom, beta
from scipy.special import polygamma
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


def generate_data(a,b, n, m):
    '''
    generates data from beta binomial distrubution

    param a: alpha_1 value
    param b: alpha_2 value
    param n: number of trials
    param m: number of samples

    return counts: the bag of words representation of data
    return samples: the generated data
    '''
    p = beta.rvs(a, b, size=m)
    # print(p)
    samples = [betabinom.rvs(n, p_val, 1-p_val, size=20) for p_val in p]
    counts = np.array([[np.sum(arr == 0), np.sum(arr == 1)] for arr in samples])
    return counts,samples

def is_positive_semi_definite(matrix):
    '''
    checks if the matrix is positive semidefinite
    '''
    # Compute the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(matrix)
    # Check if all eigenvalues are non-negative
    return np.all(eigenvalues >= 0)


def alpha_grid_crlb(alpha_grid,m):
    """
    calculates and saves the CRLB for a given grid of alpha values

    param alpha_grid: alpha grid values
    param m: number of samples to be genrated for calculating the CRLB

    returns crlb_dict: a dictionary of CRLB values saved like crlb_dict[(4.1, 4.2)] = CRLB for alpha_1 = 4.1, alpha_2 = 4.2

    """
    crlb_dict = {}

    for alpha_1 in alpha_grid:
        for alpha_2 in alpha_grid:
            print(alpha_1, alpha_2)
            # Generate parameters for the Beta distribution
            a, b = alpha_1, alpha_2
            crlb = get_crlb(m, (a, b))
            # save it in the grid    
            crlb_dict[(alpha_1, alpha_2)] = crlb[(0,0)], crlb[(1,1)]

    return crlb_dict


def get_crlb(m, alpha):
    '''
    Calculates the CRLB for a given alpha values

    param m: number of samples
    param alpha: tuple of alpha values ex: alpha = (4.1, 4.2) where alpha_1 = 4.1, alpha_2 = 4.2

    returns the calculated CRLB matrix for alpha_1 and alpha_2
    '''

    alpha_1, alpha_2 = alpha
    fim = np.zeros((len(alpha), len(alpha)))
    
    # Trigamma function is the second derivative of the log gamma function
    trigamma = lambda x: polygamma(1, x)
    second_derivative_alpha_1 = 0
    second_derivative_alpha_2 = 0
    common_term = (m * (trigamma(sum(alpha))) - (trigamma(20 + sum(alpha))))
    
    for _ in range(200):
        data, _ = generate_data(alpha_1, alpha_2, n=1, m=m)
        
        # calculating second derivative for alpha_1 and alpha_2
        second_derivative_alpha_1 += (common_term + sum(trigamma(arr[0] + alpha_1) for arr in data) - (m * trigamma(alpha_1)))
        second_derivative_alpha_2 += (common_term + sum(trigamma(arr[1] + alpha_2)for arr in data) - (m * trigamma(alpha_2))) 
        

    # constructing the matrix
    fim[0][0] =  second_derivative_alpha_1/200
    fim[1][1] =  second_derivative_alpha_2/200
    fim[0][1] = fim[1][0] = common_term
    
    
    return np.linalg.inv(np.array(fim))


def monte_carlo_mse_calculation(alpha_grid, m):
    '''
    Calculates MSE by doing 200 monte carlo simulations for every alpha values

    param alpha_grid: the grid of alpha values
    param m: number of samples

    returns mse_dict: the dictionary of stored MSE for every alpha value pair
    '''
    mse_dict = {}
    mse_alpha_1, mse_alpha_2 = 0,0
    n = 1 # number of trials

    initial_alpha_guess = (1,1)

    for alpha_1 in alpha_grid:
        for alpha_2 in alpha_grid:
            alpha = (alpha_1, alpha_2)
            print(alpha)
            mse_alpha_1, mse_alpha_2 = 0,0
            for _ in range(200):
                data , _= generate_data(alpha_1,alpha_2,n,m)
                mse_1, mse_2 = mle_minka(initial_alpha_guess,alpha,data,m)
                mse_alpha_1 += mse_1
                mse_alpha_2 += mse_2
            
            mse_dict[(alpha_1,alpha_2)] = (mse_alpha_1/40, mse_alpha_2/40)
    
    return mse_dict

def mle_minka(intial_alpha_guess,alpha, data,m):
    '''
    solves for alpha_1 and alpha_2 iteratively with minka's equation of fixed point iteration

    param initial_alpha_guess: the initial values of alpha pair
    param alpha: the true alpha values
    param data: the generated data (bag of words represenattion)
    param m: number of samples

    return (mse_1, mse_2): MSE values for alpha_1 and alpha_2 respectively

    '''
    digamma = lambda x: polygamma(0, x)
    true_alpha_1, true_alpha_2 = alpha
    alpha_old = intial_alpha_guess

    # mse_min_a, mse_min_b = 1000, 1000

    for _ in range(2000):
        alpha_1_old, alpha_2_old = alpha_old
        denom = -(m * digamma(sum(alpha_old))) + (m * (digamma(20 + sum(alpha_old))))
        
        alpha_1_num = sum(digamma(arr[0] + alpha_1_old) for arr in data) - (m * digamma(alpha_1_old))
        alpha_2_num = sum(digamma(arr[1] + alpha_2_old) for arr in data) - (m * digamma(alpha_2_old))
        
        alpha_1_new = alpha_1_old * (alpha_1_num/denom)
        alpha_2_new = alpha_2_old * (alpha_2_num/denom)

        mse_1, mse_2 = calculate_squared_error(true_alpha_1, true_alpha_2, alpha_1_new, alpha_2_new)
        # if mse_1 < mse_min_a:
        #     mse_min_a = mse_1
        # if mse_2 < mse_min_b:
        #     mse_min_b = mse_2
        # print(alpha_1_new, alpha_2_new)
        alpha_old  = (alpha_1_new, alpha_2_new)
    
    return (mse_1, mse_2)


def calculate_squared_error(true_alpha_1, true_alpha_2, estimated_alpha_1, estimated_alpha_2):
    """
    Calculate the squared error for the estimated parameters against the true parameters.

    Parameters:
    true_alpha_1 (float): The true value of alpha_1.
    true_alpha_2 (float): The true value of alpha_2.
    estimated_alpha_1 (float): The estimated value of alpha_1.
    estimated_alpha_2 (float): The estimated value of alpha_2.

    Returns:
    tuple: A tuple containing the squared errors for alpha_1 and alpha_2.
    """
    squared_error_alpha1 = (true_alpha_1 - estimated_alpha_1) ** 2
    squared_error_alpha2 = (true_alpha_2 - estimated_alpha_2) ** 2
    return squared_error_alpha1, squared_error_alpha2


def mom_estimation(data, n):
    '''
    calculates the method of moment estimation from the data

    param data: the generated data (bag of words representation)
    param n: number of trials

    '''

    # calculate the first and second moments
    m1_alpha_1 = np.mean(data[:, 0])
    m1_alpha_2 = np.mean(data[:, 1])

    m2_alpha_1 = np.mean(data[:, 0] **2)
    m2_alpha_2 = np.mean(data[:, 1] **2)

    # Calculate alpha_hat and beta_hat using the provided equations
    alpha__1_hat = (((n * m1_alpha_1) - m2_alpha_1)) / (n * ((m2_alpha_1 / m1_alpha_1) - m1_alpha_1 - 1) + m1_alpha_1)
    alpha_2_hat = (n - m1_alpha_2) * (n - (m2_alpha_2 / m1_alpha_2)) / (n * ((m2_alpha_2 / m1_alpha_2) - m1_alpha_2 - 1) + m1_alpha_2)
    
    return alpha__1_hat, alpha_2_hat 

def mom_with_mse(alpha_grid,m):
    '''
    calculates MSE for mom estimator by doing 200 monte carlo simulation

    param alpha_grid: alpha values
    param m: number of samples

    returns mom_dict: MSE calculated based on MOM estimator for every alpha value pairs

    '''
    mom_dict = {}
    for a in alpha_grid:
        for b in alpha_grid:
            print(a,b)
            mse_a, mse_b = 0, 0
            for _ in range(200):
                data, _ = generate_data(a,b,1,m)
                estimated_alpha_1, estimated_alpha_2 = mom_estimation(data, 1)
                error_a, error_b = calculate_squared_error(a, b, estimated_alpha_1, estimated_alpha_2)
                mse_a += error_a
                mse_b += error_b
            mom_dict[(a,b)] = (error_a/200, error_b/200)
    return mom_dict


def crlb_graph(is_alpha_1, crlb_dict, mse_dict, alpha_values):
    '''
    plots the alpha vs MSE and CRLB 

    param is_alpha_1: whether alpha_1 to be plotted
    param crlb_dict: the dictionary of CRLB values for alpha grid
    param mse_dict: the dictionary of MSE values for alpha grid
    param alpha_values: the alpha value pair needs to be plotted
    
    '''
    crlb_alpha_values, mse_alpha_values = [], []
    i = 1
    if is_alpha_1:
        for i in alpha_values:
            crlb_alpha_values.append(crlb_dict[(i, 4.0)][0])
            mse_alpha_values.append(mse_dict[(i, 4.0)][0])
            i = i + 1
    else:
        for i in alpha_values:
            crlb_alpha_values.append(crlb_dict[(4.0, i)][1])
            mse_alpha_values.append(mse_dict[(4.0, i)][1])
            i = i + 1

    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, crlb_alpha_values, marker='o')
    plt.plot(alpha_values, mse_alpha_values, marker='^')
    plt.title(f"Comparison of CRLB and MSE")
    if is_alpha_1:
        plt.xlabel(r"$\alpha_1$ values ($\alpha_2 = 4.0$)", fontsize=14)
    else:
        plt.xlabel(r"$\alpha_2$ values ($\alpha_1 = 4.0$)", fontsize=14)
    plt.ylabel("Values")
    plt.grid(True)
    plt.show()

def crlb_only_graph(is_alpha_1, crlb_dict, alpha_values):
    '''
    This function plots a graph for CRLB values only
    '''
    crlb_alpha_values = []
    i = 1
    if is_alpha_1:
        for i in alpha_values:
            crlb_alpha_values.append(crlb_dict[(i, 4.0)][0])
            i = i + 1
    else:
        for i in alpha_values:
            crlb_alpha_values.append(crlb_dict[(4.0, i)][1])
            i = i + 1
    # Using Seaborn for better aesthetics
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 7))
    plt.plot(alpha_values, crlb_alpha_values, marker='o', color='royalblue', markersize=8, linewidth=2)
    plt.title("CRLB Values Across Different alpha_1 Values", fontsize=16, fontweight='bold')  # Updated title
    plt.xlabel(r"$\alpha_1$ values ($\alpha_2 = 4.0$)", fontsize=14)
    plt.ylabel("CRLB Values", fontsize=14)
    plt.xticks(alpha_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.01, 0.03)  # Setting the y-axis range
    plt.show()


def mom_graph(is_alpha_1, crlb_dict, mom_dict, alpha_values):
    '''
    plots the mom graph of alpha values vs MSE and CRLB
    param is_alpha_1: whether alpha_1 to be plotted
    param mom_dict: MSE calcuated for alpha grid for mom estimator
    alpha_value: the alpha value to be plotted

    '''
    crlb_alpha_values, mse_alpha_values = [], []
    i = 1
    if is_alpha_1:
        for i in alpha_values:
            crlb_alpha_values.append(crlb_dict[(i, 4.0)][0])
            mse_alpha_values.append(mom_dict[(i, 4.0)][0])
            i = i + 1
    else:
        for i in alpha_values:
            crlb_alpha_values.append(crlb_dict[(4.0, i)][1])
            mse_alpha_values.append(mom_dict[(4.0, i)][1])
            i = i + 1


    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values[:10], crlb_alpha_values[:10], marker='o')
    plt.plot(alpha_values[:10], mse_alpha_values[:10], marker='^')
    plt.title(f"Comparison of CRLB and MSE with MoM")
    if is_alpha_1:
        plt.xlabel(r"$\alpha_1$ values ($\alpha_2 = 4.0$)", fontsize=14)
    else:
        plt.xlabel(r"$\alpha_2$ values ($\alpha_1 = 4.0$)", fontsize=14)
    plt.ylabel("Values")
    plt.grid(True)
    plt.show()


if __name__=='__main__':
    
    # defining the hyperparameters of the model
    n_trials = 1  # Number of trials in each sample
    m = 400 # Number of samples for each alpha pair

    # our choice of alpha grid values
    alpha_values = [4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, (4+1)]

    # getting the CRLB values:
    crlb_dict = alpha_grid_crlb(alpha_values, m)

    # plotting the CRLB values for alpha_1
    crlb_only_graph(True, crlb_dict, alpha_values)
    # plotting the CRLB values for alpha_2
    crlb_only_graph(False, crlb_dict, alpha_values)

    # getting the MSE values with MLE
    mse_dict = monte_carlo_mse_calculation(alpha_values,m)

    # plotting alpha_1 vs MSE and CRLB
    crlb_graph(True, crlb_dict, mse_dict, alpha_values)
    # plotting alpha_2 vs MSE and CRLB
    crlb_graph(False, crlb_dict, mse_dict, alpha_values)


    # getting MSE for MOM estimators
    mom_dict = mom_with_mse(alpha_values, m)

    # plotting alpha_1 vs CRLB and MSE
    mom_graph(True, crlb_dict, mom_dict, alpha_values)
    # plotting alpha_2 vs CRLB and MSE
    mom_graph(False, crlb_dict, mom_dict, alpha_values)
