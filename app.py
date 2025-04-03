import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai
import uuid

st.set_page_config(page_title="Studienbot", layout="wide")

# ======= FHDW + ChatGPT Style CSS =======
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #f5f7fb;
        font-family: 'Segoe UI', sans-serif;
        color: #002b5c;
    }
    .block-container {
        padding: 0 3rem;
    }
    h1 {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #002b5c;
    }
    .stButton > button {
        background-color: #002b5c;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border: none;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #004a99;
        transform: scale(1.01);
    }
    input, textarea {
        background-color: #ffffff !important;
        color: #002b5c !important;
        border: 1px solid #ccd6e0 !important;
        border-radius: 6px !important;
    }
    .stTextInput > div > div > input {
        padding: 0.6rem !important;
    }
    .chat-bubble {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #004080;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .user-bubble {
        background-color: #dbeaff;
        border-left-color: #0077cc;
    }
    </style>
""", unsafe_allow_html=True)

# ======= Secrets und Setup =======
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
AZURE_BLOB_CONN_STR = st.secrets["AZURE_BLOB_CONN_STR"]
AZURE_CONTAINER = st.secrets["AZURE_CONTAINER"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pdf_processor = PDFProcessor()
db = QdrantDB(api_key=OPENAI_API_KEY, host=QDRANT_HOST, qdrant_api_key=QDRANT_API_KEY)

# ======= Sidebar – Sessions wie bei ChatGPT =======
st.sidebar.header("\U0001F4C1 Deine Sessions")
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session = None

session_names = list(st.session_state.sessions.keys())
selected = st.sidebar.selectbox("Session auswählen", session_names + ["➕ Neue starten"])

if selected == "➕ Neue starten":
    st.session_state.active_session = None
else:
    st.session_state.active_session = selected

# ======= Hauptbereich =======
st.title("Studienbot – Frage deine Unterlagen")

with st.expander("📂 Neue PDFs prüfen & laden"):
    if st.button("🔄 PDF-Sync starten"):
        with st.spinner("Lade PDFs von Azure..."):
            pdf_paths = load_pdfs_from_blob(AZURE_BLOB_CONN_STR, AZURE_CONTAINER)
            stored_sources = db.get_stored_sources()
            new_pdfs = [p for p in pdf_paths if Path(p).name not in stored_sources]

        if new_pdfs and st.button(f"🚀 {len(new_pdfs)} neue PDFs verarbeiten"):
            with st.spinner("Verarbeite PDFs..."):
                all_chunks = []
                for path in new_pdfs:
                    chunks = pdf_processor.extract_text_chunks(path)
                    all_chunks.extend(chunks)
                db.add(all_chunks)
                st.success(f"✅ {len(all_chunks)} Chunks gespeichert.")
        else:
            st.info("Keine neuen PDFs gefunden.")

# ======= Chatverlauf und Eingabe =======

if st.session_state.active_session and st.session_state.active_session in st.session_state.sessions:
    for eintrag in st.session_state.sessions[st.session_state.active_session]:
        st.markdown(f"<div class='chat-bubble user-bubble'><strong>🧑 Frage:</strong><br>{eintrag['frage']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'><strong>🤖 Antwort:</strong><br>{eintrag['antwort']}</div>", unsafe_allow_html=True)

frage = st.text_input("", placeholder="Stelle deine Frage zur FHDW oder zu Dokumenten...")
if st.button("📤 Anfrage senden") and frage:
    if not st.session_state.active_session:
        title = frage.strip()[:50]
        st.session_state.sessions[title] = []
        st.session_state.active_session = title

    session_key = st.session_state.active_session
    resultate = db.query(frage, n=30)
    kontext = prepare_context_chunks(resultate)
    verlauf = st.session_state.sessions[session_key]

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

    st.session_state.sessions[session_key].append({"frage": frage, "antwort": antwort})
    st.rerun()

# ======= Ende =======



