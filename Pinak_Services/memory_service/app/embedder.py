import numpy as np
from typing import List, Union

class Embedder:
    def __init__(self):
        # Mock embedder for testing purposes
        self.dim = 384 # Sentence-transformers default dimension

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            return np.zeros((1, self.dim), dtype="float32")
        else:
            return np.zeros((len(text), self.dim), dtype="float32")

def get_embedder() -> Embedder:
    return Embedder()