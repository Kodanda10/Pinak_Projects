## 2025-05-14 - Fix SQL injection in DatabaseManager update_memory
**Vulnerability:** Found a SQL injection vulnerability in `DatabaseManager.update_memory` caused by building dynamic SQL query columns directly from the user-provided `updates` dictionary keys without validation.
**Learning:** In python sqlite interactions, `?` binding works for values, but column names cannot be parameterized. Using string interpolation `f"{k} = ?"` directly from user input allows attackers to inject arbitrary SQL statements if keys are maliciously crafted.
**Prevention:** Explicitly enforce that dictionary keys are valid strings and Python identifiers using `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` before constructing the dynamic SQL query.
