"""Run eyupcanakman's original issue10932 inputs independently.

The individual calls are isolated only so all reported outcomes are retained.
Source: https://github.com/numba/numba/issues/10932
"""
import json
import numpy as np
import numba
import llvmlite
from numba import njit


@njit
def int_slice_mask(a, m):
    return a[1, :, m]


@njit
def set_two_masks(x, r, c):
    x[r, c] = 99
    return x


@njit
def get_two_masks(x, r, c):
    return x[r, c]


@njit
def mask_2d(x, m):
    return x[m]


a = np.arange(24).reshape(2, 3, 4)
m = np.array([True, False, False, True])
r = np.array([True, False, True])
c = np.array([False, True, True])
x = np.arange(6).reshape(2, 3)
print(json.dumps({'runtime': {'numba': numba.__version__,
                            'llvmlite': llvmlite.__version__,
                            'numpy': np.__version__}}))
failures = 0
for fn, args in ((int_slice_mask, (a, m)),
                 (set_two_masks, (np.arange(9).reshape(3, 3), r, c)),
                 (get_two_masks, (np.arange(9).reshape(3, 3), r, c)),
                 (mask_2d, (x, x > 2))):
    expected = fn.py_func(*(arg.copy() for arg in args))
    result = {'operation': fn.py_func.__name__,
              'numpy_shape': list(expected.shape),
              'numpy_values': expected.tolist()}
    try:
        actual = fn(*(arg.copy() for arg in args))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    else:
        result.update(shape=list(actual.shape), values=actual.tolist(),
                      equal=bool(np.array_equal(actual, expected)))
    print(json.dumps(result))
    failures += not result.get('equal', False)

raise SystemExit(bool(failures))
