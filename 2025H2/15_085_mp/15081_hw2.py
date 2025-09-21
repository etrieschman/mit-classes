# %%
import numpy as np
import cvxpy as cp
import polars as pl

# %%
# read in data
problem = "problem 1"
A = np.loadtxt(f"./2025H2/15_085_mp/data/hw2_data/{problem}/A.csv", delimiter=",")
b = np.loadtxt(f"./2025H2/15_085_mp/data/hw2_data/{problem}/b.csv", delimiter=",")
c = np.loadtxt(f"./2025H2/15_085_mp/data/hw2_data/{problem}/c.csv", delimiter=",")

# %%
