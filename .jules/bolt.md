## 2024-05-23 - [Vector Store Write-Behind]
**Learning:** `MemoryService` was defeating `VectorStore`'s write-behind strategy by explicitly calling `.save()` on every write. This forced synchronous disk I/O ($O(N)$) instead of the intended throttled background save ($O(1)$).
**Action:** Always check if a storage component has a background persistence mechanism before adding explicit save calls. Trust the architectural pattern (write-behind) unless consistency requirements strictly forbid it.
