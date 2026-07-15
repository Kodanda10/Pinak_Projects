## 2024-05-24 - SQL Injection in Dynamic UPDATE Query
**Vulnerability:** SQL injection vulnerability in DatabaseManager.update_memory where dynamic dictionary keys were interpolated directly into the SET clause without validation.
**Learning:** Interpolating dictionary keys directly into the SQL string for column names allows injection of arbitrary SQL if the keys are attacker-controlled.
**Prevention:** Always validate dynamic dictionary keys used in SQL query construction using str.isidentifier() to ensure they are safe column names.
