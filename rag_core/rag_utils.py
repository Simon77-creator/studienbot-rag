import tiktoken
import openai

def prepare_context_chunks(resultate, max_tokens=6500, max_chunk_length=2000, max_per_source=4, allow_duplicates=False):
    seen_texts = set()
    if resultate and "score" in resultate[0]:
        resultate = sorted(resultate, key=lambda x: x["score"])

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    total_tokens = 0
    context_chunks = []
    source_counter = {}

    for r in resultate:
        source = r["source"]
        if source_counter.get(source, 0) >= max_per_source:
            continue

        text = r["text"][:max_chunk_length].strip()
        norm_text = text.lower().strip()
        if not allow_duplicates and norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)

        tokens = len(enc.encode(text))
        if total_tokens + tokens > max_tokens:
            break

        context_chunks.append({"text": text, "source": source, "page": r["page"]})
        total_tokens += tokens
        source_counter[source] = source_counter.get(source, 0) + 1

    return context_chunks


def build_gpt_prompt(context_chunks, frage, verlaufszusammenfassung=""):
    context = "\n\n".join([
        f"### {doc['source']} – Seite {doc['page']}\n{doc['text']}" for doc in context_chunks
    ])

    system_prompt = (
        "Du bist ein freundlicher und präziser Studienberater der FHDW.\n"
        "Nutze den folgenden Kontext und den bisherigen Gesprächsverlauf, um die Nutzerfrage zu beantworten.\n"
        f"### Verlauf (Zusammenfassung) ###\n{verlaufszusammenfassung}\n\n"
        f"### Kontext ###\n{context}\n\n### Frage ###\n{frage}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": frage}
    ]


def summarize_session_history(history, max_tokens=800, model="gpt-4o-mini", api_key=None):
    if not history or not api_key:
        return ""

    enc = tiktoken.encoding_for_model(model)
    raw_text = "\n".join([f"Frage: {e['frage']}\nAntwort: {e['antwort']}" for e in history])
    tokens = enc.encode(raw_text)
    trimmed = enc.decode(tokens[-max_tokens:])

    prompt = f"Fasse folgenden Gesprächsverlauf kompakt und sachlich zusammen:\n\n{trimmed}"

    client = openai.OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Du bist ein präziser Assistent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return res.choices[0].message.content.strip()
