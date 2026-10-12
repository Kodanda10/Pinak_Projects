## 2024-10-12 - Database Indexes
**Learning:** Adding indexes directly without checking if the specific multi-tenant columns exist fails legacy schema tests (like `test_doctor_backfill.py`), which create simplified schemas mimicking old database states.
**Action:** When adding composite indexes to multi-tenant tables, wrap `CREATE INDEX` in `self._column_exists()` checks for each column to ensure backwards compatibility with legacy schemas in tests.
