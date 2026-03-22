
## 2024-05-19 - [O(n^2) nested loop in search_hybrid]
**Learning:** In `Pinak_Services/memory_service/app/services/memory_service.py`'s `search_hybrid` function, there was an O(N*M) nested loop (where N is the combined number of semantic/episodic/procedural embeddings found and M is the result size) used to merge results from the FTS search and the vector search. Creating a dictionary for `O(1)` lookups turns this into an `O(N+M)` operation.
**Action:** Always check for nested iterations over arrays/lists when doing data mergers or combinations; converting the inner iteration array to a dictionary for key lookups is a classic O(1) fix for large sets.
