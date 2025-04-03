# app.py (überarbeitet)
import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt
from pathlib import Path
import openai

st.set_page_config(page_title="📘 Studienbot", layout="wide")

# 💼 FHDW-Styling (finale Version)
fhdw_css = """
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', Roboto, sans-serif;
    background-color: #f0f2f6;
    color: #002b5c;
}

h1 {
    font-size: 2.2rem;
    font-weight: 800;
    border-bottom: 3px solid #002b5c;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

input, textarea {
    border: 1px solid #ccc !important;
    border-radius: 6px !important;
    background-color: #fff;
    padding: 0.5rem;
}

.stButton > button {
    background-color: #002b5c;
    color: white;
    font-weight: 600;
    padding: 0.6em 1.5em;
    border-radius: 4px;
    border: none;
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    background-color: #003c85;
    transform: scale(1.02);
}

div[data-testid="stAlert"] {
    border-left: 6px solid #002b5c;
    background-color: #e6eef9;
    padding: 1rem;
    border-radius: 6px;
}

.stMarkdown {
    background-color: #ffffff;
    padding: 1.2rem;
    border-radius: 6px;
    border: 1px solid #dbe2e8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stMarkdown {
    background-color: #ffffff;
    padding: 1.2rem;
    border-radius: 6px;
    border: 1px solid #dbe2e8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stMarkdown {
    background-color: #ffffff;
    padding: 1.2rem;
    border-radius: 6px;
    border: 1px solid #dbe2e8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Ergänzung: weitere UI-Elemente */
.st-expander {
    border: 1px solid #ccd6e0;
    border-radius: 6px;
    background-color: #ffffff;
}
.st-expanderHeader {
    font-weight: 600;
    background-color: #f9fbff;
}

.stTextInput input {
    background-color: #f9f9f9 !important;
    border: 1px solid #ccc !important;
    padding: 0.6rem !important;
    border-radius: 6px !important;
    color: #002b5c !important;
}

div.stButton > button {
    background-color: #002b5c !important;
    color: white !important;
    font-weight: 600;
    border-radius: 6px !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 1rem;
}

</style>
"""

st.markdown(fhdw_css, unsafe_allow_html=True)

st.title("📘 Studienbot – Frage deine Unterlagen")

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

# ✅ Manuell neue PDFs prüfen (Lazy)
with st.expander("📂 Neue PDFs prüfen und laden"):
    if st.button("🔄 Jetzt nach neuen PDFs in Azure suchen"):
        with st.spinner("📥 Prüfe Azure auf neue PDFs..."):
            pdf_paths = load_pdfs_from_blob(AZURE_BLOB_CONN_STR, AZURE_CONTAINER)
            stored_sources = db.get_stored_sources()
            new_pdfs = [p for p in pdf_paths if Path(p).name not in stored_sources]

        if new_pdfs:
            if st.button(f"🚀 {len(new_pdfs)} neue PDFs verarbeiten"):
                with st.spinner("🔧 Verarbeite neue PDFs..."):
                    all_chunks = []
                    for path in new_pdfs:
                        chunks = pdf_processor.extract_text_chunks(path)
                        all_chunks.extend(chunks)
                    db.add(all_chunks)
                    st.success(f"✅ {len(all_chunks)} neue Abschnitte gespeichert.")
        else:
            st.info("✅ Keine neuen PDFs gefunden – alles ist aktuell.")

# 🧠 Fragestellung
frage = st.text_input("❓ Deine Frage:", placeholder="Was möchtest du wissen?")
fragen_knopf = st.button("📤 Anfrage senden")

if frage and fragen_knopf:
    with st.spinner("🧠 Studienbot denkt nach..."):
        resultate = db.query(frage, n=30)
        kontext = prepare_context_chunks(resultate)

        if not kontext:
            st.warning("❌ Keine relevanten Informationen gefunden.")
        else:
            messages = build_gpt_prompt(kontext, frage)
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
            antwort = response.choices[0].message.content
            st.markdown(antwort)

    if st.checkbox("🔎 Kontext anzeigen"):
        for c in kontext:
            st.markdown(f"**{c['source']} – Seite {c['page']}**\n\n{c['text']}\n\n---")


