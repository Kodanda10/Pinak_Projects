## 2024-05-23 - VectorStore Pre-allocation
**Learning:** `np.vstack` and `np.concatenate` inside a loop created an O(N²) bottleneck for vector insertion. Python's list appending is amortized O(1), but since we use Numpy arrays for storage, we must manually implement amortized growth (doubling capacity).
**Action:** When managing Numpy arrays that grow over time, always track capacity vs size and pre-allocate/double capacity to avoid quadratic copying costs.
