#!/usr/bin/env python
# coding: utf-8

# # Prosjekt 2 Stokmod





import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
import time


# 0  s (susceptible)
# 1  il (lightly infected)
# 2  ih (heavily infected)

class state:
    '''
    State class to easily attribute start- and end times to a specific state
    '''
    def __init__(self, startTime, currentState):
        self.startTime = startTime
        self.currentState = currentState
        self.rates = [100, 7, 20]
        
    def sojourn(self):
        '''Takes a stochastic selection of exp(A), A=lambda, mu_i, mu_h. Returns this time (endTime)'''
        self.endTime = self.startTime + np.random.exponential(self.rates[self.currentState])
        return self.endTime
    
    def jump(self):
        '''Jumps to the next state, assumes sojourn has been ran on the state'''
        if self.currentState == 0:
            if np.random.uniform() < 0.9:
                return state(self.endTime, 1)
            else:
                return state(self.endTime, 2)
        else:
            return state(self.endTime, 0)

def simCold(days=1000*365):
    '''
    Simulates the common cold for "days" days.
    Starts with a susceptible state in day 0 and a global timer. Subtracts the global timer with every state's
    sojourn time until it's zero (or below). Adds all new states in a list, whomst is returned
    '''
    start = state(0, 0)
    globalTime = days
    states = [start]
    i = 0
    while globalTime > 0:
        globalTime -= states[i].sojourn()-states[i].startTime
        states.append(states[i].jump())
        i += 1
    states[-1].sojourn()
    return states

def plotStates(states, days=5*365):
    '''
    Plots the development of the cold over "days" days.
    Input: states, list of state-objects. days (optional), number of days to be plotted.
    For each state, draws horizontal lines as long as we are in said state. 
    '''
    stateLabels = [r'S', r'$I_L$', r'$I_H$']
    for st in states:
        if st.startTime <= days:
            plt.hlines(st.currentState, st.startTime, st.endTime)
    plt.title("Realization of X(t) over {:.0f}".format(days / 365) + " years")
    plt.xlabel("Time (days)")
    plt.ylabel("State", rotation="horizontal", position=(-0.1, 1.02))
    plt.yticks((0, 1, 2), stateLabels)
    plt.xlim(right=days)
    plt.show()

def longRun(states):
    '''
    Calculates the long run proportion of how much time is spent in each of the states
    Input: states, list of state-objects.
    Initializes a dictionary with states as keys, containing the float 0.0.
    Adds the sojourn time to dictionary with corresponding key, and divides by total (all sojourn times) 
    '''
    stateD = {0: 0.0, 1: 0.0, 2 : 0.0}
    total = 0
    for st in states:
        total += st.endTime-st.startTime
        stateD[st.currentState] += st.endTime-st.startTime
    print("Long run proportion of:")
    print("Susceptible: {:.2f}".format(stateD[0]/total*100), "%")
    print("Lightly infected: {:.2f}".format(stateD[1]/total*100), "%")
    print("Highly infected: {:.2f}".format(stateD[2]/total*100), "%")



#Task 1c
def task1c():
    allStates = simCold()
    plotStates(allStates)

def task1d():
    allStates = simCold()
    longRun(allStates)



def countBetween(states, checkState=2):
    """
    Finds the average distance between the end of a heavy infection and the start of the next heavy infection.
    Input: states, list containing state-objects
    """
    firstFound = False #Flag to check if found state is first
    distances = np.zeros_like(states, dtype=np.float32)
    for i, st in np.ndenumerate(states):
        if st.currentState == checkState:
            if not firstFound:
                #Edge-case when finding the first "checkState" - can't calculate distance to previous
                firstFound = True #Toggle flag
            else:
                timerStop = st.startTime
                distances[i] = (timerStop-timerStart)
            timerStart = st.endTime
    return np.average(distances[distances != 0]) #Returns the average of all nonzero distances

def task1e():
    allStates = simCold()
    print("Average time between heavy infections: ")
    print("One realization: ", countBetween(allStates))

    N = 1000
    timeAtStart = time.time()
    counts = np.zeros(N)
    for i in range(N):
        sys.stdout.write("\rComputing average: " + str(round((i+1)/N*100, 1)) + "%   Estimated time remaining: " + str(round((time.time()-timeAtStart)*N/(i+1) - (time.time()-timeAtStart))) + " seconds        ")
        sys.stdout.flush()
        allStates = simCold()
        counts[i] = countBetween(allStates)
    sys.stdout.write("\rAverage over " + str(N)+ " realizations:  " + str(np.average(counts)) + "                                                       \n")



# TASK 2

def corr(theta1, theta2):
    """Calculates correlation for this task"""
    return (1 + 15*np.absolute(theta1-theta2))*np.exp(-15 * np.absolute(theta1-theta2))

def cov(theta1, theta2):
    """Calculates covariance using correlation relation"""
    return 0.5**2 * corr(theta1, theta2)

def covMatrix(thetas1, thetas2):
    """Produces the covariance matrix, using cov-function above"""
    sigma = np.zeros((len(thetas1), len(thetas2)))
    for i, thetai in np.ndenumerate(thetas1):
        for j, thetaj in np.ndenumerate(thetas2):
            sigma[i,j] = cov(thetai, thetaj)
    return sigma

def predInt(i, vars_C):
    """Returns 90% prediciton interval for theta_i"""
    return 1.645*np.sqrt(vars_C[i])

def computeParams(a, b, obs, mu_a, mu_b):
    """Computes mean and variance from formulae described in the project"""
    mu_c = mu_a + covMatrix(a, b) @ np.linalg.inv(covMatrix(b, b)) @ (obs-mu_b)
    sigma_c = covMatrix(a,a) - covMatrix(a,b) @ np.linalg.inv(covMatrix(b,b)) @ covMatrix(b,a)
    vars_c = np.diagonal(sigma_c)
    return mu_c, vars_c

def findBounds(mus, var):
    """Returns upper and lower bound with respect to the prediciton interval"""
    upper = np.zeros_like(mus)
    lower = np.zeros_like(mus)
    for i, mu in np.ndenumerate(mus):
        upper[i], lower[i] = mu + predInt(i, var), mu - predInt(i, var)
    return upper, lower


# In[69]:


def task2a():
    A_1        = np.linspace(0.25, 0.5, num=51, endpoint=True)
    B_1        = np.array([0.3, 0.35, 0.39, 0.41, 0.45])
    observed_1 = np.array([0.5, 0.32, 0.4, 0.35, 0.6])

    mu_A_1 = np.array(51*[0.5])
    mu_B_1 = np.array(5*[0.5])


    mu_C_1, vars_C_1 = computeParams(A_1, B_1, observed_1, mu_A_1, mu_B_1)
    upper_1, lower_1 = findBounds(mu_C_1, vars_C_1)

    plt.plot(A_1, mu_C_1, label=r'$\mu_C$')
    plt.plot(A_1, upper_1, label="Upper bound")
    plt.plot(A_1, lower_1, label="Lower bound")
    plt.plot(B_1, observed_1, 'ro', label="Observed values")
    plt.legend()
    plt.title(r"Prediction of $y(\theta)$ with 90% prediciton interval")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$y(\theta)$")
    plt.show()



# In[70]:


def task2b():
    A_1 = np.linspace(0.25, 0.5, num=51, endpoint=True)
    B_1 = np.array([0.3, 0.35, 0.39, 0.41, 0.45])
    observed_1 = np.array([0.5, 0.32, 0.4, 0.35, 0.6])

    mu_A_1 = np.array(51 * [0.5])
    mu_B_1 = np.array(5 * [0.5])

    mu_C_1, vars_C_1 = computeParams(A_1, B_1, observed_1, mu_A_1, mu_B_1)

    plt.plot(A_1, norm.cdf(0.3*np.ones_like(mu_C_1), mu_C_1, np.sqrt(vars_C_1)))
    plt.title(r"Conditional probability of $Y(\theta)<0.3$")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$Pr\{Y(\theta) < 0.3\}$")
    plt.show()



# In[71]:


def task2c():
    A_2        = np.linspace(0.25, 0.5, num=51, endpoint=True)
    B_2        = np.array([0.30, 0.33, 0.35, 0.39, 0.41, 0.45])
    observed_2 = np.array([0.50, 0.40, 0.32, 0.40, 0.35, 0.60])

    mu_A_2 = 0.5*np.ones_like(A_2)
    mu_B_2 = 0.5*np.ones_like(B_2)

    mu_C_2, vars_C_2 = computeParams(A_2, B_2, observed_2, mu_A_2, mu_B_2)

    # Due to computer rounding, we had a variance of ~ -1 * 10^-15 at one point, so we just take the absolute
    # value to mitigate this problem
    vars_C_2 = np.absolute(vars_C_2)

    upper_2, lower_2 = findBounds(mu_C_2, vars_C_2)

    fig, axs = plt.subplots(1, 2, figsize = (10.4, 3.8))

    axs[0].plot(A_2, mu_C_2, label=r'$\mu_C$')
    axs[0].plot(A_2, upper_2, label="Upper bound")
    axs[0].plot(A_2, lower_2, label="Lower bound")
    axs[0].plot(B_2, observed_2, 'ro', label="Observed values")
    axs[0].legend()
    axs[0].set_title(r"Prediction of $y(\theta)$ with 90% prediciton interval")
    axs[0].set_xlabel(r"$\theta$")
    axs[0].set_ylabel(r"$y(\theta)$")

    prob = norm.cdf(0.3*np.ones_like(mu_C_2), mu_C_2, np.sqrt(vars_C_2))

    # Due to a bug in scipy.stats.norm.cdf(), we get a zero-divison-error. However, we know that a normal distribution
    # with variance 0 is simply all the probability gathered at the mean. And because the mean at this point is
    # greater than 0.3, the cumulative probability will be zero:
    prob = np.nan_to_num(prob)

    axs[1].plot(A_2, prob)
    axs[1].set_title(r"Condtional probability of $Y(\theta) < 0.3$")
    axs[1].set_xlabel(r"$\theta$")
    axs[1].set_ylabel(r"$Pr\{Y(\theta) < 0.3\}$")
    axs[1].axvline(A_2[np.argmax(prob)], color='C1', linestyle="dashed", label=r"$\theta_{max} = " + str(A_2[np.argmax(prob)]) + "$")
    axs[1].legend()

    plt.subplots_adjust(wspace=0.3)

    plt.show()


def main():
    print("What task do you want to see? Type")
    print("1c")
    print("1d")
    print("1e")
    print("2a")
    print("2b")
    print("2c")
    print("q to quit\n")
    while True:
        np.random.seed(23)  # Note: We set a seed so that you'll get the same realization as us, comment out to reinstate randomness
        x = input()
        if x == "1c":
            task1c()
        elif x == "1d":
            task1d()
        elif x == "1e":
            task1e()
        elif x == "2a":
            task2a()
        elif x == "2b":
            task2b()
        elif x == "2c":
            task2c()
        elif x == "q":
            break
        print("\n")

main()


