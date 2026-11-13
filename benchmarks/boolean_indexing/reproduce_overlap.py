"""TheTesla's original PR10955 row-overlap input, kept as source evidence."""
import json
import numpy as np
from numba import njit


@njit
def row_overlap(x):
    x[1, 3:6] = x[1, 2:5]
    return x


x = np.array([[3, 7, 1, 0, 0, 0, 0, 0],
              [4, 3, 4, 6, 5, 0, 0, 0]])
expected = row_overlap.py_func(x.copy())
actual = row_overlap(x.copy())
print(json.dumps({'source': 'https://github.com/numba/numba/pull/10955',
                  'expected': expected.tolist(), 'actual': actual.tolist(),
                  'equal': bool(np.array_equal(actual, expected))}))
raise SystemExit(not np.array_equal(actual, expected))
