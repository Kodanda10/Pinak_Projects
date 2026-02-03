## 2024-05-23 - NumPy Array Concatenation Bottleneck
**Learning:** Repeatedly using `np.vstack` or `np.concatenate` in a loop for growing arrays creates an O(N^2) performance bottleneck because it copies the entire array content on every addition.
**Action:** Use a pre-allocated array with a capacity tracking mechanism and resize (double capacity) only when full to achieve amortized O(N) performance.
