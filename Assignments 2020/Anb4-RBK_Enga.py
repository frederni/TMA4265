import numpy as np

N = 100000
lambda_VIF = 1.2  # Number of goals per match.
lambda_RBK = 2  # Number of goals per match.
lambdamin_VIF = 1.2/90  # Number of goals per minute.
lambdamin_RBK = 2/90  # Number of goals per minute.
lambdamin_total = lambdamin_RBK + lambdamin_VIF
lambda_total = lambda_VIF + lambda_RBK

goals_VIF = np.random.poisson(1.2, N)
goals_RBK = np.random.poisson(2, N)
goals_total = goals_VIF + goals_RBK

# b

lambda_firsthalf = (lambda_VIF+lambda_RBK)/2
goals_firsthalf = np.random.poisson(lambda_firsthalf, N)
prob_zeroGoals = sum(goals_firsthalf==0)/N

# c

c = 0
for n in range(0, N):
    if goals_VIF[n] == goals_RBK[n] and goals_VIF[n] == 2:
        c = c+1

prob_22 = c/N  # Probability that the result is 2-2.

# d

firstgoal_RBK = np.random.exponential(1/lambdamin_RBK, N)
firstgoal_VIF = np.random.exponential(1/lambdamin_VIF, N)
firstgoal = np.zeros(N)

for n in range(0, N):
    if firstgoal_RBK[n]<firstgoal_VIF[n]:
        firstgoal[n] = firstgoal_RBK[n]
    else:
        firstgoal[n] = firstgoal_VIF[n]

firstgoal.mean()

# e

prob_goalsBeforeBreak=sum(np.random.poisson(lambda_VIF/3, N) > 0)/N  # The number of goals scored in the last 30 minutes of the first half does not depend on the number of goals scored in the first 15 min of the first half.