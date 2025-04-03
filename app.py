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
.stTextInput input, .stSelectbox select {
    border-radius: 6px;
    font-size: 1rem;
    padding-right: 3rem !important;
}
.chat-bubble {
    background-color: #1e293b;
    color: #f1f5f9;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    border-left: 4px solid #2563eb;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    width: fit-content;
    max-width: 85%;
    word-break: break-word;
    font-size: 1.05rem;
    line-height: 1.5;
}
.user-bubble {
    background-color: #334155;
    border-left-color: #0284c7;
    color: #f1f5f9;
    margin-left: auto;
    text-align: right;
    align-self: flex-end;
}
.bot-bubble {
    margin-right: auto;
    text-align: left;
    align-self: flex-start;
}
.sidebar .block-container {
    padding: 1rem;
}
.send-button {
    position: absolute;
    right: 0.75rem;
    top: 0.4rem;
    z-index: 10;
}
.send-button button {
    background: none;
    border: none;
    font-size: 1.3rem;
    padding: 0;
    margin: 0;
    color: #1d4ed8;
}
.input-wrapper {
    position: relative;
}
.loader {
    width: 30px;
    display: flex;
    justify-content: space-between;
    margin-top: 0.3rem;
}
.loader div {
    width: 6px;
    height: 6px;
    background: #94a3b8;
    border-radius: 50%;
    animation: blink 1.4s infinite ease-in-out both;
}
.loader div:nth-child(2) { animation-delay: 0.2s; }
.loader div:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0; }
    40% { opacity: 1; }
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
st.markdown("<p style='font-size:1.1rem; color:#1e293b; font-weight: 600;'>Dieser Chatbot hilft dir dabei, gezielt Fragen zu deinen Studienunterlagen zu stellen. Lade relevante PDFs hoch und erhalte präzise, kontextbasierte Antworten aus deinen Dokumenten.</p>", unsafe_allow_html=True)

aktive_session = st.session_state.active_session
if aktive_session and aktive_session in st.session_state.sessions:
    for eintrag in st.session_state.sessions[aktive_session]:
        st.markdown(f"<div class='chat-bubble user-bubble'><strong>👤</strong><br>{eintrag['frage']}</div>", unsafe_allow_html=True)
        if eintrag['antwort'] == "...":
            st.markdown("<div class='chat-bubble bot-bubble'><strong>🤖</strong><br><div class='loader'><div></div><div></div><div></div></div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble bot-bubble'><strong>🤖</strong><br>{eintrag['antwort']}</div>", unsafe_allow_html=True)

# Chat Input unten
with st.form(key="chat_form"):
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    frage = st.text_input("Deine Frage:", placeholder="Was möchtest du wissen?", label_visibility="visible")
    st.markdown('<div class="send-button">', unsafe_allow_html=True)
    abschicken = st.form_submit_button("➤")
    st.markdown('</div></div>', unsafe_allow_html=True)

if abschicken and frage:
    if not aktive_session:
        title = frage.strip()[:50]
        st.session_state.sessions[title] = []
        st.session_state.active_session = title
        aktive_session = title

    verlauf = st.session_state.sessions[aktive_session]
    verlauf.append({"frage": frage, "antwort": "..."})
    st.rerun()

if aktive_session and st.session_state.sessions[aktive_session]:
    last = st.session_state.sessions[aktive_session][-1]
    if last["antwort"] == "...":
        resultate = db.query(last["frage"], n=30)
        kontext = prepare_context_chunks(resultate)
        verlauf = st.session_state.sessions[aktive_session][:-1]

        verlaufszusammenfassung = summarize_session_history(
            verlauf, max_tokens=800, model="gpt-4o-mini", api_key=OPENAI_API_KEY
        )

        messages = build_gpt_prompt(kontext, last["frage"], verlaufszusammenfassung)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        antwort = response.choices[0].message.content
        st.session_state.sessions[aktive_session][-1]["antwort"] = antwort
        st.rerun()
