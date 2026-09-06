## 2024-05-24 - O(1) Numpy Appends in Vector Store
**Learning:** The previous naive Numpy vector store appended vectors one at a time using `np.vstack`. Because `np.vstack` creates a full copy of the arrays, appending N vectors incrementally becomes an O(N^2) operation, making insertion scaling very poor.
**Action:** Implemented O(1) appending by accumulating vectors into standard Python lists (buffers) first, and only flushing/concatenating them to the main Numpy arrays lazily during reads or explicit saves, yielding significant speedups.
