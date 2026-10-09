## 2026-10-09 - Fix SQL injection in dynamic UPDATE queries
**Vulnerability:** The `DatabaseManager.update_memory` dynamically built SQL statements using unsanitized user dictionary keys `set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`. This permitted attackers to bypass constraints or edit extra columns.
**Learning:** Python dictionaries derived from external inputs (e.g., via FastAPI) still require sanitization when used directly in SQL schema generation.
**Prevention:** Use `.isidentifier()` to validate that dynamically supplied keys are strictly safe valid schema identifiers. Exclude system-protected fields like `tenant`, `id`, or `project_id`.
