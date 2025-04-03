# app.py – mit ChatGPT-Style Sessions
import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai

st.set_page_config(page_title="\ud83d\udcd8 Studienbot", layout="wide")

# FHDW CSS
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
</style>
"""
st.markdown(fhdw_css, unsafe_allow_html=True)

st.title("\ud83d\udcd8 Studienbot – Frage deine Unterlagen")

# Secrets laden
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
AZURE_BLOB_CONN_STR = st.secrets["AZURE_BLOB_CONN_STR"]
AZURE_CONTAINER = st.secrets["AZURE_CONTAINER"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# Init Services
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pdf_processor = PDFProcessor()
db = QdrantDB(api_key=OPENAI_API_KEY, host=QDRANT_HOST, qdrant_api_key=QDRANT_API_KEY)

# Session-Verwaltung wie ChatGPT
st.sidebar.title("\ud83d\udc64 Deine Sessions")
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session = None

# Auswahl oder Neue Session
session_names = list(st.session_state.sessions.keys())
selected = st.sidebar.selectbox("Wähle eine Session", session_names + ["\u2795 Neue starten"])
if selected == "\u2795 Neue starten":
    st.session_state.active_session = None
else:
    st.session_state.active_session = selected

# PDF Check (manuell)
with st.expander("\ud83d\udcc2 Neue PDFs pr\u00fcfen und laden"):
    if st.button("\ud83d\udd04 Jetzt nach neuen PDFs suchen"):
        with st.spinner("\ud83d\udcc5 Pr\u00fcfe Azure auf neue PDFs..."):
            pdf_paths = load_pdfs_from_blob(AZURE_BLOB_CONN_STR, AZURE_CONTAINER)
            stored_sources = db.get_stored_sources()
            new_pdfs = [p for p in pdf_paths if Path(p).name not in stored_sources]

        if new_pdfs and st.button(f"\ud83d\ude80 {len(new_pdfs)} neue PDFs verarbeiten"):
            with st.spinner("\u2699\ufe0f Verarbeite neue PDFs..."):
                all_chunks = []
                for path in new_pdfs:
                    chunks = pdf_processor.extract_text_chunks(path)
                    all_chunks.extend(chunks)
                db.add(all_chunks)
                st.success(f"\u2705 {len(all_chunks)} neue Abschnitte gespeichert.")
        else:
            st.info("\u2705 Keine neuen PDFs gefunden.")

# Fragestellung
frage = st.text_input("\u2753 Deine Frage:", placeholder="Was möchtest du wissen?")
fragen_knopf = st.button("\ud83d\udce4 Anfrage senden")

if frage and fragen_knopf:
    if not st.session_state.active_session:
        title = frage.strip()[:50]
        st.session_state.sessions[title] = []
        st.session_state.active_session = title

    session_key = st.session_state.active_session
    with st.spinner("\ud83e\uddd0 Studienbot denkt nach..."):
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
        st.markdown(antwort)
        st.session_state.sessions[session_key].append({"frage": frage, "antwort": antwort})

if st.session_state.active_session and st.checkbox("\ud83d\udd5c Verlauf anzeigen"):
    st.markdown(f"### Verlauf: **{st.session_state.active_session}**")
    for eintrag in reversed(st.session_state.sessions[st.session_state.active_session]):
        st.markdown(f"**\ud83e\uddd1 Frage:** {eintrag['frage']}")
        st.markdown(f"**\ud83e\udd16 Antwort:** {eintrag['antwort']}")
        st.markdown("---")


