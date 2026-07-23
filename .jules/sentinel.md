## 2025-10-24 - Dynamic SQL Update Vulnerability
**Vulnerability:** SQL injection in DatabaseManager.update_memory via dynamically constructed UPDATE queries where column names derived from dictionary keys were not validated.
**Learning:** Even internal API inputs mapped directly to database columns without ORM abstraction must be strictly validated. Relying on paramaterized values is insufficient if the column names themselves are attacker-controlled.
**Prevention:** Always validate dynamic keys against an allowed list or, at minimum, strictly enforce valid identifier structures (e.g., `str.isidentifier()`) before interpolating them into SQL clauses.
