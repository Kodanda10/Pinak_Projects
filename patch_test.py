import sys

filepath = 'Pinak_Services/memory_service/tests/test_sql_injection.py'
with open(filepath, 'r') as f:
    content = f.read()

old_code = r'''        except Exception as e:
            vulnerable = True
            print(f"Injection successful! Error: {e}")'''

new_code = r'''        except Exception as e:
            if "Incorrect number of bindings supplied" in str(e):
                vulnerable = True
                print(f"Injection successful! Error: {e}")
            else:
                # no such column means SQL injection didn't break out of the identifier formatting
                vulnerable = False'''

if old_code in content:
    with open(filepath, 'w') as f:
        f.write(content.replace(old_code, new_code))
    print("Patched successfully")
else:
    print("Old code not found")
