import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai

st.set_page_config(page_title="Studienbot", layout="wide")

# Style mit Light/Dark-Mode Fixes
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.block-container { padding: 2rem 3rem; }

/* Chat-Bubble-Styles */
.chat-bubble {
    background-color: #f1f5f9;
    color: #1a1a1a;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #004080;
    word-wrap: break-word;
    max-width: 90%;
}

.user-bubble {
    background-color: #dbeafe;
    color: #1a1a1a;
    text-align: right;
    margin-left: auto;
    border-left: none;
    border-right: 4px solid #1d4ed8;
}

@media (prefers-color-scheme: dark) {
    .chat-bubble {
        background-color: #1e293b;
        color: #ffffff;
        border-left: 4px solid #3b82f6;
    }
    .user-bubble {
        background-color: #334155;
        color: #ffffff;
        border-right: 4px solid #3b82f6;
    }
}

.stTextInput input, .stSelectbox select, .stButton button {
    border-radius: 6px;
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

aktive_session = st.session_state.active_session
if aktive_session:
    st.markdown(f"# {aktive_session}")
else:
    st.title("📘 Studienbot – Frag deine Dokumente")
    st.markdown("<p style='color:#666;font-size:1rem;'>Dieser Chatbot hilft dir dabei, gezielt Fragen zu deinen Studienunterlagen zu stellen. Lade relevante PDFs hoch und erhalte präzise, kontextbasierte Antworten.</p>", unsafe_allow_html=True)

# Verlauf anzeigen
if aktive_session and aktive_session in st.session_state.sessions:
    for eintrag in st.session_state.sessions[aktive_session]:
        st.markdown(f"<div class='chat-bubble user-bubble'>{eintrag['frage']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'>{eintrag['antwort']}</div>", unsafe_allow_html=True)

# Chat Input unten
frage = st.text_input("Deine Frage:", placeholder="Was möchtest du wissen?")
if frage:
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


