# %%
import numpy as np

# HW2 2 circuits 2 ways part b
A = np.array([[-1, 0, 2], [3, -5, 0], [0, 5, 3]])
b = np.array([75, 90, 0])
x = np.linalg.solve(A, b)
print("HW2 2 circuits 2 ways part b")
print("Currents:\t", np.round(x, 2))
# %%
# HW2 circuits 2 ways circuit 2 part b
A = np.array([[7, -5, -2], [-1, 0, 2], [3, -5, 0]])
A_inv = np.linalg.inv(A)
A @ A_inv
# %%
