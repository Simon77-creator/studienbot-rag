import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai

st.set_page_config(page_title="Studienbot", layout="centered")

# ====== STYLE ======
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0f1117 !important;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    padding: 2rem 3rem;
    max-width: 768px;
    margin: auto;
}
.chat-left, .chat-right {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: inline-block;
    max-width: 80%;
    word-wrap: break-word;
}
.chat-left {
    background-color: #1e293b;
    border-left: 4px solid #2563eb;
}
.chat-right {
    background-color: #334155;
    border-right: 4px solid #2563eb;
    margin-left: auto;
    text-align: right;
}
input[type="text"] {
    padding: 0.6rem;
    border-radius: 8px;
    border: 1px solid #334155;
    background-color: #1e1e24;
    color: #fff;
}
button[kind="primary"] {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ====== SECRETS & INIT ======
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

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session = None
    st.session_state.initial_input = True
if "frage_input_clear" not in st.session_state:
    st.session_state.frage_input_clear = False

# ====== SIDEBAR ======
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/FHDW_logo_201x60.png", width=150)

with st.sidebar.expander("📂 Sitzungen verwalten"):
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

aktive_session = st.session_state.active_session

# ====== HEADLINE ======
if st.session_state.initial_input:
    st.title("📘 Studienbot – Frag deine Dokumente")
    st.markdown("""
    <p style='color:#94a3b8; font-weight:500;'>
        Dieser Chatbot hilft dir dabei, gezielt Fragen zu deinen Studienunterlagen zu stellen.
        Lade relevante PDFs hoch und erhalte präzise, kontextbasierte Antworten aus deinen Dokumenten.
    </p>
    """, unsafe_allow_html=True)
elif aktive_session:
    st.markdown(f"### 📁 {aktive_session}  | 🤖 Modell: gpt-4o-mini")

# ====== CHATVERLAUF ======
if aktive_session and aktive_session in st.session_state.sessions:
    for eintrag in st.session_state.sessions[aktive_session]:
        st.markdown(f"<div class='chat-container'><div class='chat-right'>{eintrag['frage']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-container'><div class='chat-left'>{eintrag['antwort']}</div></div>", unsafe_allow_html=True)

# ====== FRAGE-EINGABE ======
if st.session_state.frage_input_clear:
    frage_vorbelegt = ""
    st.session_state.frage_input_clear = False
else:
    frage_vorbelegt = st.session_state.get("frage_input", "")

col1, col2 = st.columns([6, 1])
with col1:
    frage = st.text_input("Deine Frage:", value=frage_vorbelegt, placeholder="Was möchtest du wissen?", key="frage_input", label_visibility="collapsed")
with col2:
    abgeschickt = st.button("➤", use_container_width=True)

# ====== SENDEN & RAG FLOW ======
if frage and (abgeschickt or frage_vorbelegt):
    if not aktive_session:
        title = frage.strip()[:50]
        st.session_state.sessions[title] = []
        st.session_state.active_session = title
        aktive_session = title

    st.session_state.initial_input = False

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
    st.session_state.frage_input_clear = True
    st.rerun()


