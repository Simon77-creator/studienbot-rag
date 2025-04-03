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
    background-color: #0f172a;
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}
.block-container { padding: 2rem 3rem; }
.stTextInput input {
    border-radius: 6px;
    background-color: #1e293b;
    color: white;
    border: none;
    padding: 0.75rem;
    width: 100%;
}
.user-bubble, .bot-bubble {
    max-width: 70%;
    padding: 1rem;
    margin: 0.5rem;
    border-radius: 12px;
    word-wrap: break-word;
    font-size: 1.05rem;
}
.user-bubble {
    background-color: #1e293b;
    color: white;
    margin-left: auto;
    text-align: right;
}
.bot-bubble {
    background-color: #1e40af;
    color: white;
    margin-right: auto;
    text-align: left;
}
.description {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: -1rem;
    margin-bottom: 2rem;
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
aktive_session = st.session_state.active_session
if aktive_session and aktive_session in st.session_state.sessions:
    verlauf = st.session_state.sessions[aktive_session]
    for eintrag in verlauf:
        st.markdown(f"<div class='user-bubble'>{eintrag['frage']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-bubble'>{eintrag['antwort']}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='description'>Dieser Chatbot hilft dir dabei, gezielt Fragen zu deinen Studienunterlagen zu stellen. Lade relevante PDFs hoch und erhalte präzise, kontextbasierte Antworten aus deinen Dokumenten.</div>", unsafe_allow_html=True)

# Eingabe
with st.form(key="frage_form"):
    frage = st.text_input("Deine Frage:", key="frage_eingabe", label_visibility="collapsed")
    submitted = st.form_submit_button("▶", use_container_width=True)

if submitted and frage:
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
    st.session_state["frage_eingabe"] = ""
    st.rerun()
