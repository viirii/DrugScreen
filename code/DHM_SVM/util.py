import numpy as np


def select(base, mask):
    assert base.size > 0
    if sum(mask) == 0:
        return np.zeros((0, base.shape[1]))
    boolean_mask = np.reshape(mask == 1, mask.size)
    assert boolean_mask.shape == (mask.shape[0], )
    result = base[boolean_mask]
    assert result.shape == (sum(mask), base.shape[1])
    return result


def stack(base, mask, to_stack_on):
    assert base.size > 0
    num_features = base.shape[1]
    assert to_stack_on.size == 0 or to_stack_on.shape[1] == num_features
    if sum(mask) == 0:
        if to_stack_on.size == 0:
            return np.zeros((0, num_features))
        return to_stack_on
    result = np.vstack((select(base, mask), to_stack_on))
    assert result.shape == (sum(mask) + to_stack_on.shape[0], num_features)
    return result
