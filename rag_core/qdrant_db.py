from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
import openai

class QdrantDB:
    def __init__(self, api_key, host, qdrant_api_key):
        self.api_key = api_key
        self.client = QdrantClient(
            url=host,
            api_key=qdrant_api_key
        )
        self.collection = "studienbot"
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def embed_text(self, text):
        client = openai.OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def add(self, documents):
        points = []
        for doc in documents:
            vector = self.embed_text(doc["content"])
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=doc["metadata"] | {"text": doc["content"]}
            ))
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, question, n=30):
        query_vector = self.embed_text(question)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=n,
            with_payload=True
        )
        return [{
            "text": h.payload["text"],
            "source": h.payload.get("source", ""),
            "page": h.payload.get("page", 0),
            "score": h.score
        } for h in hits]

    def get_stored_sources(self):
        scroll = self.client.scroll(collection_name=self.collection, with_payload=True, limit=10000)
        return list(set(item.payload.get("source", "") for item in scroll[0] if item.payload))

