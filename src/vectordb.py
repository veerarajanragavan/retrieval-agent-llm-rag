import faiss
import numpy as np

class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        self.index.add(vectors.astype("float32"))
        self.docs.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5):
        D, I = self.index.search(query_vector.astype("float32"), k)
        results = []
        for idx_list, dist_list in zip(I, D):
            hits = []
            for i, d in zip(idx_list, dist_list):
                if i < len(self.docs):
                    hits.append({"doc": self.docs[i], "score": float(d)})
            results.append(hits)
        return results
