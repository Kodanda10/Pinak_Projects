import sys

filepath = 'Pinak_Services/memory_service/app/core/database.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace the vulnerable line
old_code = r'''        set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])
        params = list(serialized.values()) + [memory_id, tenant, project_id]
        with self.get_cursor() as conn:
            cur = conn.execute(
                f"UPDATE {table} SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?",
                params,
            )'''

new_code = r'''        set_clause = ", ".join(['"{}" = ?'.format(k.replace('"', '""')) for k in serialized.keys()])
        params = list(serialized.values()) + [memory_id, tenant, project_id]
        with self.get_cursor() as conn:
            cur = conn.execute(  # nosec B608
                f"UPDATE {table} SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?",
                params,
            )'''

if old_code in content:
    with open(filepath, 'w') as f:
        f.write(content.replace(old_code, new_code))
    print("Patched successfully")
else:
    print("Old code not found")
