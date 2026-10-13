import sqlite3
import re

def check_sql_injection():
    with open('Pinak_Services/memory_service/app/core/database.py', 'r') as f:
        content = f.read()

    matches = re.finditer(r'execute\(f[\'"](.*?)[\'"]\)', content, re.DOTALL)
    for m in matches:
        print(m.group(1))

    matches = re.finditer(r'execute\(\s*f["\']{3}(.*?)["\']{3}', content, re.DOTALL)
    for m in matches:
        print(m.group(1))
