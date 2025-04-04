import streamlit as st
from rag_core.azure_loader import load_pdfs_from_blob
from rag_core.pdf_processor import PDFProcessor
from rag_core.qdrant_db import QdrantDB
from rag_core.rag_utils import prepare_context_chunks, build_gpt_prompt, summarize_session_history
from pathlib import Path
import openai

st.set_page_config(page_title="Studienbot", layout="wide")

# Theme-sicheres Styling mit Dark/Light Mode
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    padding: 0;
    margin: 0;
}

.chat-bubble {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    max-width: 85%;
    word-wrap: break-word;
    overflow-wrap: break-word;
    line-height: 1.5;
}

.user-bubble {
    margin-left: auto;
    text-align: right;
    border-left: none;
    border-right: 4px solid #2563eb;
}

@media (prefers-color-scheme: dark) {
    .chat-bubble {
        background-color: #1e293b;
        color: #ffffff;
        border-left: 4px solid #2563eb;
    }
    .user-bubble {
        background-color: #334155;
        color: #ffffff;
    }
    body {
        background-color: #0e1217;
    }
}

@media (prefers-color-scheme: light) {
    .chat-bubble {
        background-color: #f0f4f9;
        color: #000000;
        border-left: 4px solid #2563eb;
    }
    .user-bubble {
        background-color: #e4edf7;
        color: #000000;
    }
    body {
        background-color: #ffffff;
    }
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

# Sitzungsspeicher
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session = None
if "show_description" not in st.session_state:
    st.session_state.show_description = True

# Sidebar mit Logo & Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/FHDW_logo_201x60.png", width=150)
st.sidebar.markdown("## Studienbot")

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

# Titel & Beschreibung
if aktive_session:
    st.title(f"🧾 {aktive_session}")
else:
    st.title("📘 Studienbot – Frag deine Dokumente")
    if st.session_state.show_description:
        st.markdown("Dieser Chatbot hilft dir dabei, gezielt Fragen zu deinen Studienunterlagen zu stellen. Lade relevante PDFs hoch und erhalte präzise, kontextbasierte Antworten aus deinen Dokumenten.")

# Chatverlauf
if aktive_session and aktive_session in st.session_state.sessions:
    for eintrag in st.session_state.sessions[aktive_session]:
        st.markdown(f"<div class='chat-bubble user-bubble'>{eintrag['frage']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'>{eintrag['antwort']}</div>", unsafe_allow_html=True)

# Eingabe
frage = st.chat_input("Deine Frage:")
if frage:
    if not aktive_session:
        title = frage.strip()[:50]
        st.session_state.sessions[title] = []
        st.session_state.active_session = title
        st.session_state.show_description = False
        aktive_session = title

    resultate = db.query(frage, n=30)
    kontext = prepare_context_chunks(resultate)
    verlauf = st.session_state.sessions[aktive_session]

    verlaufszusammenfassung = summarize_session_history(
        verlauf, max_tokens=800, model="gpt-4o-mini", api_key=OPENAI_API_KEY
    )

    messages = build_gpt_prompt(
        context_chunks=kontext,
        frage=frage,
        verlaufszusammenfassung=verlaufszusammenfassung,
        api_key=OPENAI_API_KEY
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=1500
    )
    antwort = response.choices[0].message.content

    st.session_state.sessions[aktive_session].append({"frage": frage, "antwort": antwort})
    st.rerun()

# Optional: Kontext anzeigen
if aktive_session and st.checkbox("🔎 Kontext anzeigen"):
    for c in kontext:
        st.markdown(f"**{c['source']} – Seite {c['page']}**\n\n{c['text']}\n\n---")

