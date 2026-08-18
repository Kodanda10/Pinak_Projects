## 2024-11-20 - Fix SQL Injection in Database Layer
**Vulnerability:** SQL injection possible via unescaped dict keys dynamically used as column names in UPDATE queries.
**Learning:** Dynamic column names from user input or dictionaries can be exploited if not properly escaped, even when using parameterized queries for values.
**Prevention:** Always quote and escape column names dynamically added to the SET clause or any other part of an SQL query.
