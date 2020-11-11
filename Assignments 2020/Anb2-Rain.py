import numpy as np
import random
import matplotlib.pyplot as plt

days = 365
X_initial = 0 #0 if rain, 1 if not rain today.

X = np.zeros(days)
X[0] = X_initial
P = np.array([[0.7, 0.3], [0.25, 0.75]])

for n in range(1,days):
    u = random.uniform(0, 1)
    if u < P[int(X[n-1]), 0]:
        X[n] = 0
    else:
        X[n] = 1

raindays = np.count_nonzero(X == 0)
prob_rain = raindays/days

plt.scatter(range(1,366), X)
plt.xlabel("Days")
plt.ylabel("States")
