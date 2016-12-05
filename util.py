import numpy as np


def select(base, mask):
    if sum(mask) == 0:
        return np.array([])
    # if len(mask[mask == 1]) == 1:
    #     print(base)
    #     print(mask)
    return base[mask == 1]


def stack(base, mask, to_stack_on):
    if sum(mask) == 0:
        return to_stack_on
    return np.vstack((select(base, mask), to_stack_on))
