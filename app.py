import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai

st.set_page_config(page_title="Studienbot", layout="wide")

# Helles, funktionales Layout
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
    color: #002b5c;
}
.block-container {
    padding: 2rem 3rem;
}
input, textarea {
    background-color: #f8f9fa !important;
    color: #002b5c !important;
    border: 1px solid #ccd6e0 !important;
    border-radius: 6px !important;
    padding: 0.5rem !important;
}
.stTextInput > div > div > input {
    padding: 0.6rem !important;
}
.stButton > button {
    background-color: #002b5c;
    color: white;
    font-weight: 600;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    border: none;
}
.stButton > button:hover {
    background-color: #004a99;
}
.chat-bubble {
    background-color: #f2f6fa;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 4px solid #004080;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}
.user-bubble {
    background-color: #e4edf7;
    border-left-color: #0077cc;
}
</style>
""", unsafe_allow_html=True)

# Secrets prüfen
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

# Sidebar mit Sessions + PDF-Sync
st.sidebar.title("📚 Studienbot")

st.sidebar.subheader("📂 Deine Sessions")
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session = None

session_names = list(st.session_state.sessions.keys())
selected = st.sidebar.selectbox("Session", session_names + ["➕ Neue starten"])
if selected == "➕ Neue starten":
    st.session_state.active_session = None
else:
    st.session_state.active_session = selected

st.sidebar.subheader("⚙️ Einstellungen")
if st.sidebar.button("🔄 Neue PDFs laden"):
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
            st.sidebar.success(f"✅ {len(all_chunks)} neue Abschnitte gespeichert.")
    else:
        st.sidebar.info("📁 Keine neuen PDFs gefunden.")

# Hauptbereich
st.title("Studienbot – Frag deine Dokumente")
aktive_session = st.session_state.active_session
if aktive_session and aktive_session in st.session_state.sessions:
    for eintrag in st.session_state.sessions[aktive_session]:
        st.markdown(f"<div class='chat-bubble user-bubble'><strong>🧑 Frage:</strong><br>{eintrag['frage']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'><strong>🤖 Antwort:</strong><br>{eintrag['antwort']}</div>", unsafe_allow_html=True)

# Eingabe
with st.container():
    col1, col2 = st.columns([6, 1])
    with col1:
        frage = st.text_input("", placeholder="Deine Frage hier eingeben...")
    with col2:
        abschicken = st.button("Senden")

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




