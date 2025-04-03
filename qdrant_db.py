
# rag_core/qdrant_db.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from chromadb.utils import embedding_functions
import uuid

class QdrantDB:
    def __init__(self, api_key: str, host: str, qdrant_api_key: str, collection_name="studienbot"):
        self.embedding = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")
        self.client = QdrantClient(host=host, api_key=qdrant_api_key)
        self.collection = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        if self.collection not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def add(self, documents):
        points = []
        for doc in documents:
            vector = self.embedding.embed(doc["content"])
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": doc["content"],
                    "source": doc["metadata"]["source"],
                    "page": doc["metadata"]["page"]
                }
            )
            points.append(point)
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, question: str, n=30):
        query_vector = self.embedding.embed(question)
        hits = self.client.search(collection_name=self.collection, query_vector=query_vector, limit=n)
        return [{
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "page": hit.payload["page"],
            "score": hit.score
        } for hit in hits]
