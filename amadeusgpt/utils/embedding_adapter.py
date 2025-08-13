from typing import Union
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, provider_type: str, model_name: str = "BAAI/bge-m3", openai_model: str = "text-embedding-3-small"):
        self.provider_type = provider_type
        if self.provider_type == "openrouter":
            self.embedding_model = SentenceTransformer(model_name)
        self.embeddings = self._Embeddings(self.provider_type, self.embedding_model or openai_model)

    class _Embeddings:
        def __init__(self, provider: str, model: Union[str, SentenceTransformer]):
            self.provider = provider
            self.model = model
        
        def create(self, input: list[str], model: Union[str, SentenceTransformer]):
            if self.provider == "openrouter":
                vecs = self.model.encode(input, normalize_embeddings=True, convert_to_tensor=False)
                if isinstance(input, str):
                    vecs = [vecs]  # (1, D)

                # Convert each embedding to a plain Python list
                vecs = [v.tolist() for v in vecs]

                # Mimic OpenAI's Embedding.create() response
                return type("FakeResponse", (), {
                    "data": [
                        type("EmbeddingData", (), {
                            "embedding": emb,
                            "index": idx
                        })()
                        for idx, emb in enumerate(vecs)
                    ]
                })()
            else:
                return self.model.embeddings.create(input=input, model=model)
