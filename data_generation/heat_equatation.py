import numpy as np

# Example:
def heat_2d(idx, s=256, num_steps=50, T=50.0, **kwargs):
    initial_condition = np.zeros((s, s))
    solution = np.zeros((s, s, num_steps))
    return initial_condition, solution