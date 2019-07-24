import numpy as np
import scipy as sp
import scipy.stats as st
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

def brownian_motion_1d(sigma, T):
    """
    Simulate a 1D Brownian motion with parameter sigma^2 for T seconds. 
    
    Input:
        sigma:  sigma^2 is the parameter of the Brownian motion to be simulated. 
        T:      in seconds, the length of time of the simulation. 
    """

    k = 10000  # choose a large k, then one step is 1/k second. 
    
    X = 0
    for t in range(int(np.floor(T*k))):
        # Plot 
        plt.figure(1)
        plt.clf()
        plt.ion()  # without this, plt.show() will block the code execution
        plt.scatter(X, 0, c='red', s=50, alpha=1.0)
        plt.xlim(-3*sigma*np.sqrt(T), 3*sigma*np.sqrt(T))
        plt.ylim(-0.5, 0.5)
        plt.title('time = ' + str(t/k) + 's')
        plt.grid()    
        plt.show()        
        plt.pause(1/k)  
        
        X = X + np.random.normal(0, sigma/np.sqrt(k))
        
    
def brownian_motion_2d(sigma, T):
    """
    Simulate a 1D Brownian motion with parameter sigma^2 for T seconds. 
    
    Input:
        sigma:  sigma^2 is the parameter of the Brownian motion to be simulated. 
        T:      in seconds, the length of time of the simulation. 
    """

    k = 10000  # choose a large k, then one step is 1/k second. 
    plot_lim = 3*sigma*np.sqrt(T)
    
    X = np.zeros(2)
    for t in range(int(np.floor(T*k))):
        # Plot 
        plt.figure(1)
        plt.clf()
        plt.ion()  # without this, plt.show() will block the code execution
        plt.scatter(X[0], X[1], c='red', s=50, alpha=1.0)
        plt.xlim(-plot_lim, plot_lim)
        plt.ylim(-plot_lim, plot_lim)
        plt.title('time = ' + str(t/k) + 's')
        plt.grid()    
        plt.show()        
        plt.pause(1/k)  
        
        X = X + np.random.multivariate_normal(np.zeros(2), sigma**2/k * np.eye(2))


if __name__ == "__main__":
    # brownian_motion_1d(1, 1)
    brownian_motion_2d(1, 1)