
## 2026-03-02 - Dynamic Array VectorStore
**Learning:** Sequential np.vstack creates O(N^2) bottlenecks when inserting into Numpy arrays.
**Action:** Use pre-allocated NumPy arrays with size/capacity tracking for O(1) amortized inserts.
