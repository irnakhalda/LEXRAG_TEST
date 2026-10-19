from openai import OpenAI
import httpx
import faiss
from tqdm import tqdm
import numpy as np

class DenseRetriever:
    _MODEL_IDS = {
        "BGE-base-zh": "BAAI/bge-base-zh-v1.5",
        "Qwen2-1.5B": "iic/gte_Qwen2-1.5B-instruct",
    }
    _MODELSCOPE_IDS = {
        "BGE-base-zh": "AI-ModelScope/bge-base-zh-v1.5",
        "Qwen2-1.5B": "iic/gte_Qwen2-1.5B-instruct",
    }

    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = None

    def load_model(self, model_type):
        if model_type not in self._MODEL_IDS:
            raise ValueError(f"Unsupported model type: {model_type}")
        model_dir = self._MODEL_IDS[model_type]
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download(self._MODELSCOPE_IDS[model_type], revision="master")
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir, trust_remote_code=True)

    def _openai_embedding(self, texts: list, model_name):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url=self.base_url,
                follow_redirects=True,
            ),
        )

        response = client.embeddings.create(
            input=texts,
            model=model_name,
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings

    @staticmethod
    def _l2_normalize(embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def embed(self, texts, model_type, model_name=None, batch_size=32):
        if model_type in ["BGE-base-zh", "Qwen2-1.5B"]:
            self.load_model(model_type)
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding [{model_type}]"):
                batch = texts[i : i + batch_size]
                embeddings.extend(
                    self.model.encode(batch, normalize_embeddings=True)
                )
            return np.asarray(embeddings, dtype="float32")
        elif model_type == "openai":
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]
                batch_embeddings = [self._openai_embedding([text], model_name) for text in batch]
                embeddings.extend(batch_embeddings)
            return self._l2_normalize(embeddings)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def save_faiss(embeddings, faiss_type, save_path="index.faiss"):
        dim = embeddings.shape[1]
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")

        if faiss_type == "FlatIP":
            index = faiss.IndexFlatIP(dim)
        elif faiss_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 64)
        elif faiss_type == "IVF":
            nlist = min(128, int(np.sqrt(len(embeddings))))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(embeddings)
            index.nprobe = min(8, nlist // 4)
        else:
            raise ValueError(f"Unsupported FAISS type: {faiss_type}")

        index.add(embeddings)
        faiss.write_index(index, save_path)
