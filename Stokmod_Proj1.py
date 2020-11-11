# -*- coding: utf-8 -*-
"""StokMod Project 1.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Problem 1

## d)


# This functions makes a transition from one state to another given the three probabilities [beta, gamma, alpha]
def transition(state, probabilities):
    randNum = np.random.uniform()
    if randNum > probabilities[state-1]:
        return state
    else:
        return state%3 + 1

# Simulates for 20 years
def simulate(probabilities):
    realization = [1]
    for day in range(18250):
        realization.append(transition(realization[-1], probabilities))
    return realization

# Calculates average time to go from one state to another
def expected(frm, to, realization):
    algState = "Counting"
    hasTransfered = False
    start = 0
    counts = [] # This stores all the numbers of days it took to get from state 'frm' to state 'to'

    # Loop through all days
    for n in range(1, len(realization)):

        # The state of the loop: either counting days until reaching desired state, or waiting to get to the start state to start a new count
        if algState == "Counting":

            # Making sure a transfer has happened, so that a direct transfer from susceptible to susceptible is not counted as a cycle
            if realization[n] != realization[n-1]:
                hasTransfered = True
            
            # If destination state has been reached and a transfer has been made
            if realization[n] == to and hasTransfered:
                # Add number of days to counts, and reset algorithm
                counts.append(n-start)
                hasTransfered = False
                algState = "Waiting"

            # If frm == to, the loop shouldn't wait to get to the start state, because it is already in it
            if frm == to and algState == "Waiting":
                algState = "Counting"
                start = n
        
        # If the loop is in waitmode
        else:
            # Check if it should go into countmode
            if realization[n] == frm:
                algState = "Counting"
                start = n
    
    # Return the average of all counts
    return sum(counts)/len(counts)            




# Print results
def oneD():
    realization = simulate([0.05, 0.1, 0.01])
    d = [["S","I",expected(1,2, realization), 20],
         ["S","R",expected(1,3, realization), 30],
         ["S","S",expected(1,1, realization), 130]]
    df = pd.DataFrame(d,columns=[" From "," To "," Simulated ", " Analytical "])
    print("                Expected time [days]")
    print(df.to_string(index=False, justify="center"))




##f)


def plot_temporal(S, I, R):
    plt.plot(S, label="Susceptible individuals")
    plt.plot(I, label="Infected individuals")
    plt.plot(R, label="Recovered individuals")
    plt.vlines(50, 0, 1000, label="t=50", linestyle="dashed")
    plt.legend()
    plt.xlabel("t [days]")
    plt.ylabel("Number of individuals")
    plt.show()



def SimSIR(n):
    S = 950
    I = 50
    R = 0
    N = 1000
    alpha = 0.01
    gamma = 0.1
    
    SS = [S]
    II = [I]
    RR = [R]
    
    for day in range(1,n):
        ItR = np.random.binomial(I, gamma)
        I -= ItR
        R += ItR
        beta = 0.5*I/N
        StI = np.random.binomial(S, beta)
        S -= StI
        I += StI
        RtS = np.random.binomial(R, alpha)
        R -= RtS
        S += RtS
        SS.append(S)
        II.append(I)
        RR.append(R)
    return SS, II, RR

def oneF():
    Sn, In, Rn = SimSIR(300)
    plot_temporal(Sn, In, Rn)

#g)
def oneG():
    Sn, In, Rn = SimSIR(36500)
    print("S", Sn[-1], "=", round(Sn[-1]/1000*100,2), "%")
    print("I", In[-1], "=", round(In[-1]/1000*100,2), "%")
    print("R", Rn[-1], "=", round(Rn[-1]/1000*100,2), "%")

"""##h)"""

def outbreakSIR(days, sim):  # Number of days and number of simulations
  I_max_list = []
  I_min_arg_list = []
  for i in range(0, sim):
    Sn, In, Rn = SimSIR(300)
    I_max_list.append(max(In))
    I_min_arg_list.append(np.argmax(In))  # min is not needed because np.argmax takes the first index of the top value if there is multiple
  E_I_max = np.average(I_max_list)
  E_I_min_argmax = np.average(I_min_arg_list)
  return E_I_max, E_I_min_argmax

def oneH():
    E_In_max, E_In_min_argmax = outbreakSIR(300, 1000)  # 300 days and 1000 simulations
    print("Expected maximum infected:", E_In_max, ", Expected first day of most infected:", E_In_min_argmax)

# Problem 2

# a)


def simPoisson():
    n = 1000  # Number of simulations
    days = np.zeros(n)
    for i in range(101):
        days += np.random.exponential(1/1.5, n)  # The time between events in a poisson process is exponentially distributed
    successes = (days <= 59*np.ones(days.shape))  # Checks if it took less than 60 days to get 100 claims
    print("Average success rate for", n, "simulations: ", sum(successes)/len(successes))


def JointSimulation(end, lam):
    prettyColors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for n in range(10):
        nb_events = np.random.poisson(lam * end)  # number of claims by March 1st simulated by a Poisson distribution
        locations = np.random.uniform(0, end, nb_events)  # simulate the time of each event with a uniform distribution
        locations = np.sort(locations)  # we sort the times in increasing order
        events = np.linspace(0, nb_events, nb_events)  # y axis for our plot
        # plt.step(locations, events)
        for it in range(len(locations) - 1):
            plt.hlines(y=events[it], xmin=locations[it], xmax=locations[it + 1], color=prettyColors[n])
    plt.xlabel("t [days]")
    plt.ylabel("x(t)")
    plt.show()



def twoA():
    simPoisson()
    JointSimulation(59, 1.5)

"""##b)"""


def twoB(realizations=1000):  # Reusing code, can tidy up if code prettiness matters for the grade(i dont think it does)
    allZ = []
    for i in range(realizations):  # 1000 times
        Z = 0
        Xt = np.random.poisson(59*1.5)  # X(t=59) ~ Poisson(lambda t)
        for i in range(int(Xt)):  # int cast is probability unnecessary
            Z = Z + np.random.exponential(1/10)  # Cumulates from i=0,1,.., Xt
        allZ.append(Z)  # Append total claim amount to "masterlist" storing every realization
    E_Z = np.average(allZ)  # Expected total claim amount by 59 days
    Var_Z = np.var(allZ, ddof=1)  # ddof=1 means empirical variance
    return E_Z, Var_Z


def main():
    print("What task do you want to see? Type")
    print("1d")
    print("1f")
    print("1g")
    print("1h")
    print("2a")
    print("2b")
    print("q to quit\n")
    while True:
        np.random.seed(1) # Note: We set a seed so that you'll get the same realization as us, comment out to reinstate randomness
        x = input()
        if x == "1d":
            oneD()
        elif x == "1f":
            oneF()
        elif x == "1g":
            oneG()
        elif x == "1h":
            oneH()
        elif x == "2a":
            twoA()
        elif x == "2b":
            E_Z, Var_Z = twoB()
            print("Expected total claim amount: E[Z] =", E_Z, "  Variance of total claim amount: Var[Z] =", Var_Z)
        elif x == "q":
            break
        print("\n")



main()