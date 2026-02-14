## 2025-05-15 - [Ineffective Schema Validation]
**Vulnerability:** Schema validation logic was present but its return value (errors) was ignored, allowing invalid data to persist.
**Learning:** Calling a validator function is not enough; one must explicitly handle the result (raise exception or return error). Defensive coding requires checking return values.
**Prevention:** Ensure all validation calls are followed by a conditional check that halts execution on failure. Use type systems or linters that warn about unused return values if possible.
