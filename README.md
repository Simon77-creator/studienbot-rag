# 📘 Studienbot RAG (Streamlit + Qdrant + Azure)

Ein cloud-basiertes Retrieval-Augmented Generation System mit:
- ✅ Azure Blob PDF Import
- ✅ Qdrant Vektordatenbank
- ✅ OpenAI Embeddings & GPT-4o
- ✅ Streamlit UI für einfache Fragenbeantwortung

## 🚀 Deployment auf Streamlit Cloud

1. Dieses Repository forken oder klonen
2. In Streamlit Cloud deployen: https://streamlit.io/cloud
3. In den App-Einstellungen unter `secrets` folgendes eintragen:

```
OPENAI_API_KEY = "sk-..."
AZURE_BLOB_CONN_STR = "DefaultEndpointsProtocol=...=="
AZURE_CONTAINER = "chat-memory"
QDRANT_HOST = "https://<dein-cluster>.qdrant.tech"
QDRANT_API_KEY = "<dein_qdrant_key>"
```

4. App starten und PDFs laden ➜ Fragen stellen ✅
