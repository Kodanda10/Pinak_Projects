import os

file_path = "Pinak_Services/memory_service/app/services/vector_store.py"
with open(file_path, "r") as f:
    content = f.read()

old_code = r'''    def add_vectors(self, vectors: np.ndarray, ids: List[int]):'''
new_code = r'''    def add_vectors(self, vectors: np.ndarray, ids: List[int] = None):'''

# Ah wait, the review mentioned `add` method, maybe I should check `grep "def add"`?
