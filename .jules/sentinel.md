## 2024-10-24 - Mass Assignment in Dynamic Updates
**Vulnerability:** Dynamic column generation in `update_memory` allowed updating arbitrary columns by supplying manipulated keys.
**Learning:** The database schema columns must be explicitly validated and whitelist-checked prior to inclusion in the `SET` clause.
**Prevention:** Strictly validate keys in dictionary updates against allowed schemas to prevent mass assignment and unauthorized data modifications.
