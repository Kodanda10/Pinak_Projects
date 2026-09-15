import re
print(re.match(r"^[a-zA-Z0-9_]+$", "valid_col123"))
print(re.match(r"^[a-zA-Z0-9_]+$", "invalid col"))
print(re.match(r"^[a-zA-Z0-9_]+$", "invalid=1"))
print(re.match(r"^[a-zA-Z0-9_]+$", "invalid;--"))
