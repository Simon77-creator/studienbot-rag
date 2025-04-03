import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai

st.set_page_config(page_title="Studienbot", layout="wide")

# Style
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #f8fafc !important;
    color: #0f172a;
    font-family: 'Segoe UI', sans-serif;
}
.block-container { padding: 1rem 2rem; max-width: 1000px; margin: auto; }
.stTextInput input, .stSelectbox select, .stButton button {
    border-radius: 6px;
    font-size: 1rem;
}
.chat-bubble {
    background-color: #1e293b;
    color: white;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    border-left: 4px solid #2563eb;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.user-bubble {
    background-color: #334155;
    border-left-color: #0284c7;
    color: white;
}
.sidebar .block-container {
    padding: 1rem;
}
/* Send button styling */
.send-button button {
    background: none;
    border: none;
    font-size: 1.5rem;
    padding: 0 0.5rem;
    color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

# Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
AZURE_BLOB_CONN_STR = st.secrets.get("AZURE_BLOB_CONN_STR")
AZURE_CONTAINER = st.secrets.get("AZURE_CONTAINER")
QDRANT_HOST = st.secrets.get("QDRANT_HOST")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")

if not all([OPENAI_API_KEY, AZURE_BLOB_CONN_STR, AZURE_CONTAINER, QDRANT_HOST, QDRANT_API_KEY]):
    st.error("❌ Fehlende API-Zugänge oder Secrets.")
    st.stop()

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pdf_processor = PDFProcessor()
db = QdrantDB(api_key=OPENAI_API_KEY, host=QDRANT_HOST, qdrant_api_key=QDRANT_API_KEY)

# Sidebar
st.sidebar.markdown("# 📘 Studienbot")

with st.sidebar.expander("📂 Sitzungen verwalten", expanded=True):
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
        st.session_state.active_session = None

    session_names = list(st.session_state.sessions.keys())
    selected = st.selectbox("Session auswählen:", session_names + ["➕ Neue starten"])
    if selected == "➕ Neue starten":
        st.session_state.active_session = None
    else:
        st.session_state.active_session = selected

with st.sidebar.expander("⚙️ Einstellungen"):
    if st.button("📥 Neue PDFs laden"):
        with st.spinner("Lade PDFs von Azure..."):
            pdf_paths = load_pdfs_from_blob(AZURE_BLOB_CONN_STR, AZURE_CONTAINER)
            stored_sources = db.get_stored_sources()
            new_pdfs = [p for p in pdf_paths if Path(p).name not in stored_sources]

        if new_pdfs:
            with st.spinner("Verarbeite PDFs..."):
                all_chunks = []
                for path in new_pdfs:
                    chunks = pdf_processor.extract_text_chunks(path)
                    all_chunks.extend(chunks)
                db.add(all_chunks)
                st.success(f"✅ {len(all_chunks)} neue Chunks gespeichert.")
        else:
            st.info("📁 Keine neuen PDFs gefunden.")

# Hauptbereich
st.title("📘 Studienbot – Frag deine Dokumente")
st.markdown("<p style='font-size:1.1rem; color:#334155;'>Dieser Chatbot hilft dir dabei, gezielt Fragen zu deinen Studienunterlagen zu stellen. Lade relevante PDFs hoch und erhalte präzise, kontextbasierte Antworten aus deinen Dokumenten.</p>", unsafe_allow_html=True)

aktive_session = st.session_state.active_session
if aktive_session and aktive_session in st.session_state.sessions:
    for eintrag in st.session_state.sessions[aktive_session]:
        st.markdown(f"<div class='chat-bubble user-bubble'><strong>👤 Frage:</strong><br>{eintrag['frage']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'><strong>🤖 Antwort:</strong><br>{eintrag['antwort']}</div>", unsafe_allow_html=True)

# Chat Input unten
frage_col, send_col = st.columns([10, 1])
with frage_col:
    frage = st.text_input("Deine Frage:", placeholder="Was möchtest du wissen?")
with send_col:
    abschicken = st.button("➤", key="send_button")

if abschicken and frage:
    if not aktive_session:
        title = frage.strip()[:50]
        st.session_state.sessions[title] = []
        st.session_state.active_session = title
        aktive_session = title

    resultate = db.query(frage, n=30)
    kontext = prepare_context_chunks(resultate)
    verlauf = st.session_state.sessions[aktive_session]

    verlaufszusammenfassung = summarize_session_history(
        verlauf, max_tokens=800, model="gpt-4o-mini", api_key=OPENAI_API_KEY
    )

    messages = build_gpt_prompt(kontext, frage, verlaufszusammenfassung)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=1500
    )
    antwort = response.choices[0].message.content

    st.session_state.sessions[aktive_session].append({"frage": frage, "antwort": antwort})
    st.rerun()
