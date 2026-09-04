## 2024-05-18 - List-based buffering for numpy vector store
**Learning:** In standard Numpy arrays, `np.vstack` copies the entire array into a new chunk of memory each time it is called. For a vector store with continuous, small additions, this becomes an O(N) operation per addition which completely bottlenecks large memory sets.
**Action:** Use an O(1) append strategy with standard python lists (e.g. `list.append()`), and lazily flush them into the main Numpy array via `np.vstack` only when a read operation (like search or save) actually requires it.
