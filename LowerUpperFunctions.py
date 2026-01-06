import numpy as np
import matplotlib.pyplot as plt

# Define the triangular fuzzy number function
def triangular_fuzzy_number(a, b, c, x):
    if a <= x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return (c - x) / (c - b)
    else:
        return 0

# Define the Gaussian fuzzy set function
def gaussian_fuzzy_set(x, sigma, c):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# Define the generalized bell function
def bell_function(x, a, b, c):
    return 1 / (1 + (np.abs((x - c) / a) ** (2 * b)))
###############################

def F_A_upT(params, x_values, f_values):
    F_A_up = []
    for a, b, c in params:
        products = np.array([triangular_fuzzy_number(a, b, c, x) * f_x for x, f_x in zip(x_values, f_values)])
        F_A_up.append(np.max(products))
    return F_A_up

def F_A_downT(params, x_values, f_x_values):
    F_A_down = []

    for a, b, c in params:
        # Compute the ratio f(x) / triangular_fuzzy_number(a, b, c, x)
        ratios = np.array([
            f_x_values[i] / triangular_fuzzy_number(a, b, c, x_values[i])
            if triangular_fuzzy_number(a, b, c, x_values[i]) > 0 else np.inf
            for i in range(len(x_values))
        ])
        
        # Append the minimum ratio
        F_A_down.append(np.min(ratios))
    
    return F_A_down
######################################
def F_A_upG(centers, sigma, x_values,f_x_values):
    F_A_down = []
    # Compute F_A^downarrow for each center
    # Compute F_A^\uparrow for each Gaussian fuzzy set
    F_A_up = []

    # Compute F_A^\uparrow
    for c in centers:
        products = np.array([gaussian_fuzzy_set(x, sigma, c) * f_x for x, f_x in zip(x_values, f_x_values)])
        F_A_up.append(np.max(products))
    
    return F_A_up


def F_A_downG(centers, sigma, x_values,f_x_values):
    F_A_down = []
    # Compute F_A^downarrow for each center
    for c in centers:
        # Compute ratios for all x_values
        ratios = np.array([
            f_x_values[i] / gaussian_fuzzy_set(x_values[i], sigma, c)
            if gaussian_fuzzy_set(x_values[i], sigma, c) > 0 else np.inf
            for i in range(len(x_values))
        ])
        
        # Append the minimum ratio
        F_A_down.append(np.min(ratios))
    
    return F_A_down
##########################################

def F_A_downB(centers, a,b, x_values,f_x_values):
    # Compute F_A^\downarrow for each generalized bell function
    F_A_down = []
    # Compute F_A^\downarrow
    for c in centers:
        ratios = np.array([f_x / bell_function(x, a, b, c) if bell_function(x, a, b, c) > 0 else np.inf for x, f_x in zip(x_values, f_x_values)])
        F_A_down.append(np.min(ratios))
    
    return F_A_down

def F_A_upB(centers, a,b, x_values,f_x_values):
    # Compute F_A^\uparrow for each generalized bell function
    F_A_up = []

    # Compute F_A^\uparrow
    for c in centers:
        products = np.array([bell_function(x, a, b, c) * f_x for x, f_x in zip(x_values, f_x_values)])
        F_A_up.append(np.max(products))

    return F_A_up

############################################3
def f_A_downT(x,F_A_down,params):
    values = [triangular_fuzzy_number(a, b, c, x) * F_A for (a, b, c), F_A in zip(params, F_A_down)]
    return max(values)

# Define the inverse upward transformation function
def f_A_upT(x,F_A_up,params):
    ratios = [F_A / triangular_fuzzy_number(a, b, c, x) if triangular_fuzzy_number(a, b, c, x) > 0 else np.inf 
              for (a, b, c), F_A in zip(params, F_A_up)]
    return min(ratios)
##################################
# Define the inverse downward transformation function
def f_A_downB(x,F_A_down,centers,a,b):
    values = [bell_function(x, a, b, c) * F_A for c, F_A in zip(centers, F_A_down)]
    return max(values)
# Define the inverse upward transformation function
def f_A_upB(x,F_A_up,centers,a,b):
    ratios = [F_A / bell_function(x, a, b, c) if bell_function(x, a, b, c) > 0 else np.inf 
              for c, F_A in zip(centers, F_A_up)]
    return min(ratios)
########################################
# Define the inverse upward transformation function
def f_A_upG(x,F_A_up,sigma,centers):
    ratios = [F_A / gaussian_fuzzy_set(x, sigma, c) if gaussian_fuzzy_set(x, sigma, c) > 0 else np.inf 
              for c, F_A in zip(centers, F_A_up)]
    return min(ratios)
# Define the inverse downward transformation function
def f_A_downG(x,F_A_down,sigma,centers):
    values = [gaussian_fuzzy_set(x, sigma, c) * F_A for c, F_A in zip(centers, F_A_down)]
    return max(values)
######################################3
# Function to plot all Gaussian fuzzy sets
def plot_gaussian_fuzzy_sets(centers, sigma):
    x_values = np.linspace(-2.5, 2.5, 500)
    plt.figure(figsize=(10, 6))

    for i, c in enumerate(centers, start=1):
        fuzzy_values = gaussian_fuzzy_set(x_values, sigma, c)
        plt.plot(x_values, fuzzy_values, linewidth=2, label=f'$A_{{{i}}}$')

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('Membership Degrees', fontsize=16)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend(fontsize=16)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig("Gauss.pdf", format='pdf')
    plt.show()

##############################################
# Function to plot all fuzzy sets
def plot_triangular_fuzzy_sets(params):
    x_values = np.linspace(-2.5, 2.5, 500)
    plt.figure(figsize=(10, 6))

    for i, (a, b, c) in enumerate(params, start=1):
        fuzzy_values = np.array([triangular_fuzzy_number(a, b, c, x) for x in x_values])
        plt.plot(x_values, fuzzy_values, linewidth=2, label=f'$A_{{{i}}}$')

    # Optional title
    # plt.title('Triangular Fuzzy Sets')

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('Membership Degrees', fontsize=16)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend(fontsize=16)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig("Trian.pdf", format='pdf')
    plt.show()
##################################
# Function to plot all Gaussian fuzzy sets
def plot_bell_fuzzy_sets(x_values, centers, a, b):
    plt.figure(figsize=(10, 6))

    for i, c in enumerate(centers, start=1):  # start=1 for A_1, A_2, ...
        fuzzy_values = bell_function(x_values, a, b, c)
        plt.plot(x_values, fuzzy_values, linewidth=2, label=f'$A_{{{i}}}$')

    # Optional title
    # plt.title('Bell Fuzzy Sets')

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('Membership Degrees', fontsize=16)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend(fontsize=16)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig("Bell.pdf", format='pdf')
    plt.show()
