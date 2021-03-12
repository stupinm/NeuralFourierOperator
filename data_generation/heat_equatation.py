import numpy as np

# Example:
def heat_2d(idx, s=128, t=100, T=50.0, **kwargs):
    initial_condition = np.random.rand(s, s)
    solution = np.random.rand(s, s, t)
    return initial_condition, solution