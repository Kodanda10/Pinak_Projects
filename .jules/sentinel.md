## 2024-05-24 - Mass Assignment SQL Injection in update_memory
**Vulnerability:** The `update_memory` function dynamically built SQL queries from arbitrary dictionary keys provided by the caller (`updates.items()`), leading to a critical mass assignment SQL injection risk.
**Learning:** Even if values are parameterized, constructing SQL query clauses (like `SET key = ?`) directly from uncontrolled input keys allows attackers to manipulate the query structure or access/modify unauthorized columns.
**Prevention:** Always strictly validate dictionary keys used for dynamic query construction to ensure they are valid strings and Python identifiers (`key.isidentifier()`), or strictly check them against an allowed list of safe columns.
