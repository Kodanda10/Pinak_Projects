## 2024-04-18 - Optimize VectorStore batch addition
**Learning:** Calling `np.vstack` repeatedly inside a batch addition loop degrades performance to O(N^2) due to constant reallocation and copying of the underlying NumPy array.
**Action:** Use thread-local storage `threading.local()` to aggregate additions into a temporary Python list within a context manager. Perform a single `np.vstack` and `np.concatenate` on context exit.
