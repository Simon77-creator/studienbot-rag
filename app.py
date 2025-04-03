import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai

st.set_page_config(page_title="Studienbot", layout="wide")

# Custom CSS
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0d1117;
    color: #e6f0ff;
    font-family: 'Segoe UI', sans-serif;
}
.block-container { padding: 2rem 3rem; }
.description {
    font-size: 1.05rem;
    font-weight: 500;
    color: #e6f0ff;
    margin-top: -1rem;
    margin-bottom: 2rem;
}
.chat-bubble {
    background-color: #1b2a40;
    color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    max-width: 85%;
    word-wrap: break-word;
}
.chat-left {
    border-left: 4px solid #0077cc;
    align-self: flex-start;
    margin-right: auto;
}
.chat-right {
    border-right: 4px solid #3399ff;
    align-self: flex-end;
    margin-left: auto;
    background-color: #2a4365;
}
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.chat-input-container {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1.5rem;
}
.chat-input-container input {
    flex: 1;
    padding: 0.75rem;
    font-size: 1rem;
    border-radius: 6px;
    border: 1px solid #ccc;
    background-color: #0d1117;
    color: white;
}
.chat-input-container button {
    background-color: #004080;
    color: white;
    padding: 0.75rem 1.2rem;
    border: none;
    border-radius: 6px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Secrets laden
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
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/FHDW_logo_201x60.png", width=150)
st.sidebar.markdown("## Studienbot")

with st.sidebar.expander("📂 Sitzungen verwalten"):
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
    if st.button("🔄 Neue PDFs laden"):
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
st.markdown('<div class="description">Dieser Chatbot hilft dir dabei, gezielt Fragen zu deinen Studienunterlagen zu stellen. Lade relevante PDFs hoch und erhalte präzise, kontextbasierte Antworten aus deinen Dokumenten.</div>', unsafe_allow_html=True)
aktive_session = st.session_state.active_session
if aktive_session and aktive_session in st.session_state.sessions:
    for eintrag in st.session_state.sessions[aktive_session]:
        st.markdown(f"<div class='chat-container'><div class='chat-bubble chat-right'>{eintrag['frage']}</div><div class='chat-bubble chat-left'>{eintrag['antwort']}</div></div>", unsafe_allow_html=True)

# Chat Input
frage = st.text_input("Deine Frage:", placeholder="Was möchtest du wissen?", label_visibility="collapsed", key="frage_input")
absenden = st.button("➤", key="senden_btn")

if absenden and frage:
    aktive_session = st.session_state.active_session
    if not aktive_session:
        title = frage.strip()[:50]
        st.session_state.sessions[title] = []
        st.session_state.active_session = title
        aktive_session = title

    resultate = db.query(frage, n=30)
    kontext = prepare_context_chunks(resultate)
    verlauf = st.session_state.sessions[aktive_session]
    verlaufszusammenfassung = summarize_session_history(verlauf, max_tokens=800, model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    messages = build_gpt_prompt(kontext, frage, verlaufszusammenfassung)
    with st.spinner("..."):
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
    antwort = response.choices[0].message.content
    st.session_state.sessions[aktive_session].append({"frage": frage, "antwort": antwort})
    st.rerun()

