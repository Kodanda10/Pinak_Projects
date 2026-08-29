import os
import re

file_path = "Pinak_Services/memory_service/app/services/vector_store.py"
with open(file_path, "r") as f:
    content = f.read()

# I need to find the add_vectors function and check its ID generation logic.
# Wait, let me re-read the review output:
# "In the `add` method, if IDs are not provided, it generates them using `len(self.ids)`:"
# But `add_vectors` has `ids: List[int]` and doesn't generate them!
