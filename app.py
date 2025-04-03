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
    background-color: #0f1117 !important;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    padding: 2rem 3rem;
    max-width: 900px;
    margin: auto;
}
.stTextInput input, .stSelectbox select {
    border-radius: 6px;
    padding: 0.5rem;
    background-color: #1e2130;
    color: white;
}
.chat-bubble {
    background-color: #1c2a44;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 4px solid #0066cc;
    display: inline-block;
    max-width: 80%;
    word-wrap: break-word;
}
.user-bubble {
    background-color: #263349;
    margin-left: auto;
    border-left: none;
    border-right: 4px solid #0066cc;
    text-align: right;
    display: inline-block;
    max-width: 80%;
    word-wrap: break-word;
}
.chat-input-container {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.chat-input-container input {
    flex: 1;
    padding: 0.75rem;
    border-radius: 6px;
    background-color: #1e2130;
    color: white;
    border: 1px solid #333;
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
aktive_session = st.session_state.active_session
if aktive_session and aktive_session in st.session_state.sessions:
    if len(st.session_state.sessions[aktive_session]) == 0:
        st.markdown("<p style='color:#ccc;font-size:1.05rem;'>Dieser Chatbot hilft dir dabei, gezielt Fragen zu deinen Studienunterlagen zu stellen. Lade relevante PDFs hoch und erhalte präzise, kontextbasierte Antworten aus deinen Dokumenten.</p>", unsafe_allow_html=True)
    for eintrag in st.session_state.sessions[aktive_session]:
        st.markdown(f"<div class='chat-bubble user-bubble'>{eintrag['frage']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'>{eintrag['antwort']}</div>", unsafe_allow_html=True)

# Chat Input
frage = st.text_input("Deine Frage:", placeholder="Was möchtest du wissen?", label_visibility="collapsed")
abschicken = frage and st.session_state.get("letzte_frage") != frage

if abschicken:
    st.session_state["letzte_frage"] = frage
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

    with st.spinner("Antwort wird generiert..."):
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
