import numpy as np


def select(base, mask):
    assert base.size > 0
    if sum(mask) == 0:
        return np.zeros((base.shape[0], 0))
    result = base[mask == 1]
    assert result.shape == (base.shape[0], sum(mask))
    return result


def stack(base, mask, to_stack_on):
    assert base.size > 0
    x = base.shape[0]
    assert to_stack_on.size == 0 or to_stack_on.shape[0] == x
    if sum(mask) == 0:
        if to_stack_on.size == 0:
            return np.zeros((x, 0))
        return to_stack_on
    result = np.vstack((select(base, mask), to_stack_on))
    assert result.shape == (x, sum(mask) + to_stack_on.shape[1])
    return result
