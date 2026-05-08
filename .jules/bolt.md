## 2024-05-09 - [O(n^2) scaling of numpy array extensions in loops]
**Learning:** Incrementally appending directly to numpy arrays (e.g. `np.vstack`) scales quadratically with the total number of items, which can make things very slow if there's a lot of insertions.
**Action:** When performing lots of incremental additions, it's significantly faster to build a python list of the pieces to insert, and then do a single `np.vstack` or `np.concatenate` at the end to assemble the final array.
