# %%
import numpy as np
import cvxpy as cp

# %%
# FREE PROBLEM 1
# APPROACH: Formulate the problem as an optimization, with the goal of
# minimizing the deviation to the provided voltages, constrained on the new
# voltages needing to adhere to KVL. I formulate this as a mixed-integer problem
# to constrain the number of components changed to exactly 2. Code is below
# and solution is to the right -->

# SETUP: Constants
names = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"])
# voltages
v = np.array([4, 4, -1, 5, 3, 2, 2, 5, -3, 0, 4, 5]) * 1.0
# loop definitions (assigment matrix of nodes to loops, only 5 considered)
L = np.array(
    [
        #       A   B   C   D   E   F   G   H   I   J   K   L
        [1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0],  # I quad.
        [0, -1, 0, 1, -1, 0, -1, 0, 0, 0, 0, 0],  # II quad.
        [0, 0, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1],  # III quad.
        [0, 0, 0, 0, 0, -1, 0, 1, -1, 0, -1, 0],  # IV quad.
        [1, -1, 1, 0, -1, 0, 0, 1, 0, -1, -1, 1],  # Full
    ]
)
# arbitrary cap on max deviation in voltage
V_max = np.abs(v).sum() * 100

# SETUP: variables
x = cp.Variable(len(v))  # the deviation in voltage needed
c = cp.Variable(len(v), integer=True)  # binary constraint to fix changes to 2

# SETUP: objective
obj = cp.Minimize(cp.norm1(x))  # minimize voltage deviations needed

# SETUP: constraints
constraints = [
    c >= 0,
    c <= 1,  # MILP constraint
    x <= V_max * c,
    x >= -V_max * c,  # link voltage deviations to MILP
    cp.sum(c) == 2,  # Limit the number of changes to 2
    L @ (v + x) == 0,  # KVL
]

# SOLVE PROBLEM
prob = cp.Problem(obj, constraints)
prob.solve()
print("Components:\t", names)
print("V_init:\t\t", np.round(v, 2))
print("X:\t\t", np.round(x.value, 2))
print("V_new:\t\t", np.round(v + x.value, 2))

# %%
# FREE PROBLEM 2
A = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, -2, 0, 0],
        [0, 0, 1, 0, 0, 0, -6, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, -1, 0, 0, 0, 0, 0],
        # [1,1,0,-1,0,0,0,0],
        [0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ]
)
b = np.array([12, 0, 0, 2, 0, 0, 0, 0])
np.linalg.solve(A, b)

# %%
