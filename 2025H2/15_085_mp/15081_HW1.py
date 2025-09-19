# %%
import numpy as np
import cvxpy as cp
import polars as pl

pl.Config.set_tbl_rows(100)
# %%
# TEXT PROBLEM 1
# Constants
d = np.array([5, 1, 3, 1, 3, 8, 9, 8, 9, 2, 5, 3]) * 1.0
c_1 = 4
c_2 = 10

# Variables
x = cp.Variable(len(d))
delta_x_pos = cp.Variable(len(d))
delta_x_neg = cp.Variable(len(d))
I = cp.Variable(len(d))

# Constraints
constraints = [
    I[0] == 0,  # no starting inventory
    I[1:] == I[:-1] + x[1:] - d[1:],  # inventory balance
    x[0] >= d[0],  # meet first month demand
    I >= 0,  # no negative inventory
    x >= 0,  # no negative orders
    delta_x_pos >= 0,  # no negative pos changes
    delta_x_neg >= 0,  # no negative neg changes
    (delta_x_pos - delta_x_neg)[1:] == x[1:] - x[0:-1],  # link changes to orders
]

# Objective
obj_fn = c_1 * cp.sum(I[:-1]) + c_2 * cp.sum(delta_x_pos + delta_x_neg)
obj = cp.Minimize(obj_fn)

# Solve
prob = cp.Problem(obj, constraints)
prob.solve()

# Results
print("Total cost:\t", np.round(prob.value, 2))
print("Results:")
results = pl.DataFrame(
    {
        "month": np.arange(1, len(d) + 1),
        "demand": d,
        "production": np.round(x.value, 2),
        "delta_prod": np.round((delta_x_pos - delta_x_neg).value, 2),
        "inventory": np.round(I.value, 2),
    }
)
print(results)
# %%
