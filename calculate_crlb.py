import numpy as np
from scipy.stats import betabinom, beta
from scipy.special import polygamma


def calculate_crlb(data, m, alpha):
    '''
    Calculates the FIM for a given beta-binomial distribution and alpha values

    param data:
    param m:
    param alpha_1:
    param alpha_2:

    return (fim_alpha_1, fim_alpha_2)
    '''

    alpha_1, alpha_2 = alpha
    fim = np.zeros(2,2)
    
    # Trigamma function is the second derivative of the log gamma function
    trigamma = lambda x: polygamma(1, x)

    # calculating the term common to every calculation
    common_term = (m ** trigamma(sum(alpha))) - (trigamma(20+ sum(alpha)))

    # calculating second derivative for alpha_1 and alpha_2
    second_derivative_alpha_1 = common_term + sum(data + alpha_1) - (m ** trigamma(alpha_1))
    second_derivative_alpha_2 = common_term + sum(data + alpha_2) - (m ** trigamma(alpha_2))

    # common term is going to be second derivative for alpha_1, alpha_2

    # constructing the matrix
    fim[0][0] = second_derivative_alpha_1
    fim[1][1] = second_derivative_alpha_2
    fim[0][1] = fim[1][0] = common_term

    crlb = np.linalg.inv(fim)

    return crlb





