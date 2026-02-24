## 2025-05-15 - [Unsafe Deserialization in VectorStore]
**Vulnerability:** The `VectorStore` class used `numpy.load(..., allow_pickle=True)` to load vector indices. This allows arbitrary code execution if an attacker can tamper with the index file (e.g., via a separate file upload vulnerability or compromised storage).
**Learning:** `numpy.save` uses pickle for object arrays (like dictionaries). Storing `{'vectors': ..., 'ids': ...}` as a single object triggers this.
**Prevention:** Use `numpy.savez` or `numpy.savez_compressed` to store multiple arrays in a `.npz` container, which allows loading with `allow_pickle=False`. Always avoid pickling unless absolutely necessary and the source is trusted.
