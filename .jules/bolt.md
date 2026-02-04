# Bolt's Performance Journal

## 2025-05-20 - NumPy Array Growth Performance Trap
**Learning:** Using `np.vstack` and `np.concatenate` for adding vectors in a loop creates an O(N^2) complexity due to repeated full-array copying. This is a common pattern when migrating from list-based prototyping to NumPy production code.
**Action:** Always implement a dynamic array strategy (pre-allocation + tracking size/capacity) or use `list.append` and convert to NumPy only when necessary. In `VectorStore`, we implemented a doubling capacity strategy to achieve amortized O(1) insertion.
