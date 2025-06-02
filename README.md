# EXP-NO : 04 VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.
## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## POLICY ITERATION ALGORITHM
Value iteration is a method of computing an optimal MDP policy and its value.
It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation.
The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.
The algorithm is as follows:

Initialize the value function V(s) arbitrarily for all states s.
Repeat until convergence:
Initialize aaction-value function Q(s, a) arbitrarily for all states s and actions a.
For all the states s and all the action a of every state:
Update the action-value function Q(s, a) using the Bellman equation.
Take the value function V(s) to be the maximum of Q(s, a) over all actions a.
Check if the maximum difference between Old V and new V is less than theta, where theta is a small positive number that determines the accuracy of estimation.
If the maximum difference between Old V and new V is greater than theta, then
Update the value function V with the maximum action-value from Q.
Go to step 2.
The optimal policy can be constructed by taking the argmax of the action-value function Q(s, a) over all actions a.
Return the optimal policy and the optimal value function.

## VALUE ITERATION FUNCTION
### PROGRAM
```py
DEVELOPED BY : SHALINI K
REGISTER NUMBER : 212222240095

desc=['FSFH','FFFH','FHGF','HFFH']
env = gym.make('FrozenLake-v1',desc=desc)
init_state = env.reset()
goal_state = 10
P = env.env.P
```
```py
def value_iteration(P, gamma=1.0, theta=1e-10):
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```

## OUTPUT:

![image](https://github.com/user-attachments/assets/ef2b328c-b20c-4a46-84c6-2ad6f5c20927)

![image](https://github.com/user-attachments/assets/45239892-cd37-44aa-bcba-7e008bf001c3)




## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
