# Boolean advanced-indexing reproduction

Source-only commit: `afda5e2ed89a90d86cf0e57ea936b9e623614734`, based on Numba main `cdfb64df2388e1eedceb9e212e2843ea6edb263c`.

The implementation converts masks to integer coordinates once, then uses shared advanced-index broadcasting. It repairs output-axis placement, paired-mask reads and writes, multidimensional masks, newaxis/ellipsis grouping and overlap bounds. Coordinate conversion precedes assignment so a mask may alias the target.

Original report and inputs: eyupcanakman's #10932. Shared integer machinery: kc611's #10432. Physical overlap repair builds on TheTesla's #10955. Eyupcanakman's #10931 separately validates mask dimensions and parallel error behavior; integration must reconcile that carrier. The valid parallel-assignment check here does not establish all of #10931's parallel error handling.

## Recorded execution

CPython 3.12.14, NumPy 2.3.5, Numba 0.68.0 and llvmlite 0.50.0. Exact changed current-main Python modules ran over the released native runtime with unmodified current `numba/misc/heapsort.py`. No dependency-version check was changed. This was not a full development-runtime build.

`final-native.log` records nine passing methods covering the original failures, mixed broadcast masks/integer arrays, multidimensional masks, strided/read-only masks, target aliasing, row overlap, valid parallel assignment and invalid dimensions. Normal paths use `MemoryLeakMixin`; error-path allocation accounting is outside these observations. `final-original-cases.jsonl` and `overlap-after.json` retain actual original-input outputs. No speedup is claimed.

With matching development dependencies, run from the repository root:

```sh
python benchmarks/boolean_indexing/reproduce_10932.py
python benchmarks/boolean_indexing/reproduce_overlap.py
python -m unittest numba.tests.test_fancy_indexing.TestBooleanFancyIndexing numba.tests.test_fancy_indexing.TestBooleanFancyIndexErrors -v
```
