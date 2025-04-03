import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai
import uuid

st.set_page_config(page_title="Studienbot", layout="wide")

# 🎨 FHDW-inspiriertes dunkles Design
fhdw_css = """
<style>
html, body, [class*="css"]  {
    background-color: #0f1117;
    color: #f0f4fc;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #ffffff;
    font-weight: 800;
    font-size: 2.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #004080;
    margin-bottom: 1.5rem;
}
.stButton > button {
    background-color: #004080;
    color: white;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    border: none;
    border-radius: 5px;
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    background-color: #0059b3;
    transform: scale(1.02);
}
input, textarea, .stTextInput, .stTextArea {
    background-color: #1c1f26 !important;
    color: #f0f4fc !important;
    border: 1px solid #004080 !important;
    border-radius: 4px !important;
}
details {
    background-color: #1c1f26;
    color: white;
    border: 1px solid #004080;
    border-radius: 5px;
    padding: 0.4rem;
}
.stMarkdown {
    background-color: #14161c;
    color: #e0ecff;
    border-left: 4px solid #004080;
    padding: 1rem;
    margin-top: 1rem;
    border-radius: 6px;
}
[data-testid="stAlert"] {
    background-color: #182030;
    border-left: 6px solid #0077cc;
    color: #f0f4fc;
}
hr {
    border: none;
    border-top: 1px solid #2a2e39;
    margin: 1.5rem 0;
}
</style>
"""
st.markdown(fhdw_css, unsafe_allow_html=True)

st.title("Studienbot – Frage deine Unterlagen")

# 🔐 Secrets laden
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
AZURE_BLOB_CONN_STR = st.secrets["AZURE_BLOB_CONN_STR"]
AZURE_CONTAINER = st.secrets["AZURE_CONTAINER"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# 🔧 Services
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pdf_processor = PDFProcessor()
db = QdrantDB(api_key=OPENAI_API_KEY, host=QDRANT_HOST, qdrant_api_key=QDRANT_API_KEY)

# 💬 Session Management
st.sidebar.title("🗂️ Deine Sessions")
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session = None

session_names = list(st.session_state.sessions.keys())
selected = st.sidebar.selectbox("Session auswählen", session_names + ["➕ Neue starten"])

if selected == "➕ Neue starten":
    st.session_state.active_session = None
else:
    st.session_state.active_session = selected

# 📂 PDF-Upload Expander
with st.expander("📂 Neue PDFs prüfen und laden"):
    if st.button("🔄 Jetzt nach neuen PDFs suchen"):
        with st.spinner("📥 Lade PDFs aus Azure..."):
            pdf_paths = load_pdfs_from_blob(AZURE_BLOB_CONN_STR, AZURE_CONTAINER)
            stored_sources = db.get_stored_sources()
            new_pdfs = [p for p in pdf_paths if Path(p).name not in stored_sources]

        if new_pdfs and st.button(f"🚀 {len(new_pdfs)} neue PDFs verarbeiten"):
            with st.spinner("⚙️ Verarbeite PDFs..."):
                all_chunks = []
                for path in new_pdfs:
                    chunks = pdf_processor.extract_text_chunks(path)
                    all_chunks.extend(chunks)
                db.add(all_chunks)
                st.success(f"✅ {len(all_chunks)} Abschnitte gespeichert.")
        else:
            st.info("✅ Keine neuen PDFs vorhanden.")

# ❓ Frage stellen
frage = st.text_input("❓ Deine Frage:", placeholder="Was möchtest du wissen?")
fragen_knopf = st.button("📤 Anfrage senden")

if frage and fragen_knopf:
    if not st.session_state.active_session:
        title = frage.strip()[:50]
        st.session_state.sessions[title] = []
        st.session_state.active_session = title

    session_key = st.session_state.active_session
    with st.spinner("🧠 Studienbot denkt nach..."):
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

# 📜 Verlauf anzeigen
if st.session_state.active_session and st.checkbox("🕘 Verlauf anzeigen"):
    st.markdown(f"### Verlauf: **{st.session_state.active_session}**")
    for eintrag in reversed(st.session_state.sessions[st.session_state.active_session]):
        st.markdown(f"**🧑 Frage:** {eintrag['frage']}")
        st.markdown(f"**🤖 Antwort:** {eintrag['antwort']}")
        st.markdown("---")


