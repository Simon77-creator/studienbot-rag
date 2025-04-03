import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt
from pathlib import Path
import openai

st.set_page_config(page_title="📘 Studienbot RAG", layout="wide")
st.title("📘 Studienbot: Frage deine Studienunterlagen")

# 🔐 Secrets laden
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
AZURE_BLOB_CONN_STR = st.secrets["AZURE_BLOB_CONN_STR"]
AZURE_CONTAINER = st.secrets["AZURE_CONTAINER"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# 🔧 Setup
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pdf_processor = PDFProcessor()
db = QdrantDB(api_key=OPENAI_API_KEY, host=QDRANT_HOST, qdrant_api_key=QDRANT_API_KEY)

# 📂 Azure PDFs laden und checken
with st.spinner("🔍 Prüfe Azure auf neue PDFs..."):
    pdf_paths = load_pdfs_from_blob(AZURE_BLOB_CONN_STR, AZURE_CONTAINER)
    stored_sources = db.get_stored_sources()
    new_pdfs = [p for p in pdf_paths if Path(p).name not in stored_sources]

if new_pdfs:
    if st.button(f"📥 {len(new_pdfs)} neue PDFs erkannt – jetzt verarbeiten"):
        with st.spinner("🚀 Verarbeite neue PDFs..."):
            all_chunks = []
            for path in new_pdfs:
                chunks = pdf_processor.extract_text_chunks(path)
                all_chunks.extend(chunks)
            db.add(all_chunks)
            st.success(f"✅ {len(all_chunks)} neue Abschnitte gespeichert.")
else:
    st.info("✅ Es gibt keine neuen PDFs – deine Datenbank ist aktuell.")

st.caption(f"Aktuell erkannte Dateien: {len(pdf_paths)} im Azure-Container, {len(stored_sources)} in Qdrant.")

# 🧠 Frage stellen
frage = st.text_input("❓ Deine Frage:", placeholder="Was steht zur Praxisphase in den Dokumenten?")

if frage:
    with st.spinner("Antwort wird generiert..."):
        resultate = db.query(frage, n=30)
        kontext = prepare_context_chunks(resultate)

        if not kontext:
            st.warning("Keine relevanten Informationen gefunden.")
        else:
            messages = build_gpt_prompt(kontext, frage)
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # dein Modell
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
            antwort = response.choices[0].message.content
            st.markdown(antwort)

    if st.checkbox("🔎 Kontext anzeigen"):
        for c in kontext:
            st.markdown(f"**{c['source']} – Seite {c['page']}**\n\n{c['text']}\n\n---")

