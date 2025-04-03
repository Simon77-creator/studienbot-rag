
# rag_core/rag_utils.py
from collections import defaultdict
import tiktoken
import openai

def prepare_context_chunks(resultate, max_tokens=6500, max_chunk_length=2000, max_per_source=4, allow_duplicates=False):
    seen_texts = set()
    resultate = sorted(resultate, key=lambda x: -x["score"])
    enc = tiktoken.encoding_for_model("gpt-4")
    total_tokens = 0
    context_chunks = []
    source_counter = defaultdict(int)

    for r in resultate:
        source = r["source"]
        if source_counter[source] >= max_per_source:
            continue

        text = r["text"][:max_chunk_length].strip()
        if len(text) < 50:
            continue

        norm_text = text.lower().strip()
        if not allow_duplicates and norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)

        tokens = len(enc.encode(text))
        if total_tokens + tokens > max_tokens:
            break

        context_chunks.append({"text": text, "source": source, "page": r["page"]})
        total_tokens += tokens
        source_counter[source] += 1

    return context_chunks

def build_gpt_prompt(context_chunks, frage):
    context = "\n\n".join([
        f"### {doc['source']} – Seite {doc['page']}\n{doc['text']}" for doc in context_chunks
    ])

    system_prompt = (
        "Du bist ein freundlicher und präziser Studienberater der FHDW.\n"
        "Nutze ausschließlich den folgenden Kontext, um die Nutzerfrage zu beantworten.\n"
        "Wenn relevante Informationen enthalten sind, fasse sie vollständig, korrekt und strukturiert zusammen.\n"
        "Wenn keine passende Information im Kontext vorhanden ist, sage das ehrlich.\n"
        "Strukturiere deine Antwort klar: Absätze, Aufzählungen, ggf. Zwischenüberschriften.\n"
        "Zitiere wichtige Begriffe oder Formulierungen wörtlich, wenn möglich.\n"
        "Verwende Emojis nur, wenn es zur besseren Lesbarkeit beiträgt.\n\n"
        f"### Kontext ###\n{context}\n\n### Aufgabe ###\nFrage: {frage}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": frage}
    ]
