# %%
import numpy as np
import cvxpy as cp
import os
from typing import Tuple
import matplotlib.pyplot as plt

# %%
# read in data
dir = os.getcwd()
def get_data(problem):
    A = np.loadtxt(f"{dir}/data/hw2_data/{problem}/A.csv", delimiter=",")
    b = np.loadtxt(f"{dir}/data/hw2_data/{problem}/b.csv", delimiter=",")
    c = np.loadtxt(f"{dir}/data/hw2_data/{problem}/c.csv", delimiter=",")
    return A, b, c

# %%
# Precompute answer to verify my code
def cvxpy_solve(A, b, c):
    x = cp.Variable(A.shape[1], nonneg=True)
    constraints = [A@x == b]
    obj = cp.Minimize(c@x)
    problem = cp.Problem(obj, constraints)
    problem.solve()
    print("CVXPY solution:")
    print("\tstatus:\t\t", problem.status)
    print("\topt. value:\t", np.round(problem.value, 4) if problem.status == cp.OPTIMAL else None)
    return problem.status, x.value

# %%
# Simplex with a given BFS
def simplex_w_bfs(
        A, b, c, x_bfs, eps = 1e-10, max_iter = 200
    ) -> Tuple[str, np.ndarray | None, np.ndarray | None]:
    x_i = x_bfs
    x_is = [x_i.copy()]
    for i in range(max_iter):
        # -1- step 1: initialize variables
        B = x_i > 0
        A_B_inv = np.linalg.inv(A[:, B])
        u = A_B_inv @ A
        # -2- step 2: calculate cost difference
        c_bar = c - c[B] @ u
        c_bar[np.abs(c_bar) < eps] = 0
        # CHECK: return optimal if cbar > 0
        if np.all(c_bar >= 0):
            return "optimal", x_i, np.array(x_is)
        # -3- step 3: choose entering variable
        e = int(np.where(c_bar < 0)[0][0]) # first negative
        e = int(np.argmin(c_bar)) # most negative
        d = np.zeros(x_i.shape[0])
        d[e] = 1
        d[B] = -u[:, e]
        # CHECK: return unbounded if d >= 0
        if np.all(d >= 0):
            return "unbounded", None, None
        # -4- step 4: compute greedy step size
        S = (d < 0) & B
        theta = np.min(x_i[S] / -d[S])
        x_i += theta * d
        x_i[np.abs(x_i) < eps] = 0
        x_is += [x_i.copy()]
    return "max_iter", x_i, np.array(x_is)


# %%
# Two-phase simplex
def simplex_2phase(A, b, c) -> Tuple[str, np.ndarray | None, np.ndarray | None]:
    # -A- Initialize Phase 1
    A_p = np.concatenate([A, np.eye(A.shape[0])], axis=1)
    A_p[b < 0, :A.shape[1]] *= -1
    b_p = np.abs(b)
    c_p = np.concat([np.zeros(c.shape[0]), np.ones(b.shape[0])])
    xy_bfs = np.concat([np.zeros(A.shape[1]), b_p])
    # check:
    assert np.allclose(A_p @ xy_bfs, b_p)
    
    # -B- Phase 1
    print("Phase 1 simplex - find BFS")
    status, xy_opt, __ = simplex_w_bfs(A_p, b_p, c_p, xy_bfs)
    print(f"\tstatus:\t\t {status}")
    print(f"\topt. value:\t {np.round(c_p @ xy_opt, 4) if xy_opt is not None else None}")
    # check:
    if (c_p @ xy_opt) > 1e-5:
        print("Phase 2 simplex skipped - infeasible")
        print("\tstatus:\t\t infeasible")
        print("\topt. value:\t None")
        return "infeasible", None, None

    # -C- Phase 2
    x_bfs = xy_opt[:A.shape[1]]
    print("Phase 2 simplex - optimize")
    status, x_opt, x_is = simplex_w_bfs(A, b, c, x_bfs)
    print(f"\tstatus:\t\t {status}")
    print(f"\topt. value:\t {np.round(c @ x_opt, 4) if x_opt is not None else None}")
    return status, x_opt, x_is

# %%
# Run all problems
for problem in ["problem 1", "problem 2", "problem 3"]:
    print(f"==== {problem} ====")
    A, b, c = get_data(problem)
    # status_cvxpy, x_cvxpy = cvxpy_solve(A, b, c)
    print("---- part 1 ----")
    status_simplex, x_simplex, x_is = simplex_2phase(A, b, c)
    print("---- part 2 ----")
    if x_is is not None:
        gap = (x_is - x_simplex) @ c
        plt.plot(np.arange(1, x_is.shape[0]+1), gap)
        plt.title(f"Optimal gap, {problem}")
        plt.xlabel("k")
        plt.ylabel("c^T(x_k - x*)")
    else:
        print("No iterations to show")
    print("\n")

# %%
