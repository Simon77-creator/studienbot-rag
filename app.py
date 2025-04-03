import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai
import uuid

st.set_page_config(page_title="Studienbot", layout="wide")

# FHDW-Styling
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
.stMarkdown {
    background-color: #ffffff;
    padding: 1.2rem;
    border-radius: 6px;
    border: 1px solid #dbe2e8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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

# 🔧 Services initialisieren
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pdf_processor = PDFProcessor()
db = QdrantDB(api_key=OPENAI_API_KEY, host=QDRANT_HOST, qdrant_api_key=QDRANT_API_KEY)

# 🧠 Session-Handling wie bei ChatGPT
st.sidebar.title("👤 Deine Sessions")
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session = None

session_names = list(st.session_state.sessions.keys())
selected = st.sidebar.selectbox("Wähle eine Session", session_names + ["➕ Neue starten"])

if selected == "➕ Neue starten":
    st.session_state.active_session = None
else:
    st.session_state.active_session = selected

# 📂 PDF Check
with st.expander("📂 Neue PDFs prüfen und laden"):
    if st.button("🔄 Jetzt nach neuen PDFs suchen"):
        with st.spinner("📥 Prüfe Azure auf neue PDFs..."):
            pdf_paths = load_pdfs_from_blob(AZURE_BLOB_CONN_STR, AZURE_CONTAINER)
            stored_sources = db.get_stored_sources()
            new_pdfs = [p for p in pdf_paths if Path(p).name not in stored_sources]

        if new_pdfs and st.button(f"🚀 {len(new_pdfs)} neue PDFs verarbeiten"):
            with st.spinner("⚙️ Verarbeite neue PDFs..."):
                all_chunks = []
                for path in new_pdfs:
                    chunks = pdf_processor.extract_text_chunks(path)
                    all_chunks.extend(chunks)
                db.add(all_chunks)
                st.success(f"✅ {len(all_chunks)} neue Abschnitte gespeichert.")
        else:
            st.info("✅ Keine neuen PDFs gefunden.")

# ❓ Nutzerfrage stellen
frage = st.text_input("❓ Deine Frage:", placeholder="Was möchtest du wissen?")
fragen_knopf = st.button("📤 Anfrage senden")

if frage and fragen_knopf:
    if not st.session_state.active_session:
        session_title = frage.strip()[:50]
        st.session_state.sessions[session_title] = []
        st.session_state.active_session = session_title

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

        st.session_state.sessions[session_key].append({
            "frage": frage,
            "antwort": antwort
        })

# 🕘 Verlauf anzeigen
if st.session_state.active_session and st.checkbox("🕘 Verlauf anzeigen"):
    st.markdown(f"### Verlauf: **{st.session_state.active_session}**")
    for eintrag in reversed(st.session_state.sessions[st.session_state.active_session]):
        st.markdown(f"**🧑 Frage:** {eintrag['frage']}")
        st.markdown(f"**🤖 Antwort:** {eintrag['antwort']}")
        st.markdown("---")


