from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import openai
import uuid

class QdrantDB:
    def __init__(self, api_key: str, host: str, qdrant_api_key: str, collection_name="studienbot"):
        self.api_key = api_key
        self.client = QdrantClient(
            url=host,
            api_key=qdrant_api_key,
        )
        self.collection = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if self.collection not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def embed_text(self, text: str):
        client = openai.OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",  # bleibt gleich
            input=text
        )
        return response.data[0].embedding

    def add(self, documents):
        points = []
        for doc in documents:
            vector = self.embed_text(doc["content"])
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
        query_vector = self.embed_text(question)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=n
        )
        return [{
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "page": hit.payload["page"],
            "score": hit.score
        } for hit in hits]

    def get_stored_sources(self):
        result, _ = self.client.scroll(
            collection_name=self.collection,
            limit=10000,
            with_payload=True
        )
        return set(p.payload["source"] for p in result if "source" in p.payload)

