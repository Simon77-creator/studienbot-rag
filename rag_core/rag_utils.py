import tiktoken
 import openai
 
 def prepare_context_chunks(resultate, max_tokens=8000, max_chunk_length=2000, max_per_source=5, allow_duplicates=False):
 def prepare_context_chunks(
     resultate,
     max_tokens=8000,
     max_chunk_length=2000,
     max_per_source=5,
     allow_duplicates=False,
     keyword_boost=["bachelor", "master", "spezialisierung", "vertiefung", "modul"]
 ):
     seen_texts = set()
     if resultate and "score" in resultate[0]:
         resultate = sorted(resultate, key=lambda x: x["score"])
 
     enc = tiktoken.encoding_for_model("gpt-4o-mini")
     total_tokens = 0
     context_chunks = []
     source_counter = {}
 
     def boost_score(text):
         lowered = text.lower()
         return sum(1 for kw in keyword_boost if kw in lowered)
 
     if resultate and "score" in resultate[0]:
         for r in resultate:
             r["boost"] = boost_score(r["text"])
         resultate = sorted(resultate, key=lambda x: (x["score"] - x["boost"] * 0.1))
 
     for r in resultate:
         source = r["source"]
         if source_counter.get(source, 0) >= max_per_source:
 @@ -26,31 +39,54 @@ def prepare_context_chunks(resultate, max_tokens=8000, max_chunk_length=2000, ma
         if total_tokens + tokens > max_tokens:
             break
 
         context_chunks.append({"text": text, "source": source, "page": r["page"]})
         context_chunks.append({
             "text": text,
             "source": source,
             "page": r["page"]
         })
         total_tokens += tokens
         source_counter[source] = source_counter.get(source, 0) + 1
 
     return context_chunks
 
 def detect_question_type(frage: str) -> str:
     frage_lower = frage.lower()
     if any(kw in frage_lower for kw in ["unterschied", "vergleich", "vs", "besser als", "besser geeignet"]):
         return "vergleich"
     elif any(kw in frage_lower for kw in ["welche", "was für", "optionen", "auswahl", "spezialisierungen", "gibt es"]):
         return "auswahl"
     elif any(kw in frage_lower for kw in ["was ist", "erkläre", "definition", "bedeutet"]):
         return "definition"
     else:
         return "allgemein"
 
 def build_gpt_prompt(context_chunks, frage, verlaufszusammenfassung=""):
     context = "\n\n".join([
     frage_typ = detect_question_type(frage)
 
     typ_prompt = {
         "vergleich": "Wenn sich die Frage auf einen Vergleich bezieht, stelle Gemeinsamkeiten und Unterschiede strukturiert dar.",
         "auswahl": "Wenn mehrere Optionen möglich sind, liste sie klar auf und beschreibe sie stichpunktartig.",
         "definition": "Wenn die Frage nach einer Erklärung oder Definition verlangt, erkläre präzise und sachlich.",
         "allgemein": "Beantworte die Frage klar und vollständig, so sachlich wie möglich."
     }
 
     kontext = "\n\n".join([
         f"### {doc['source']} – Seite {doc['page']}\n{doc['text']}" for doc in context_chunks
     ])
 
     system_prompt = (
         "Du bist ein freundlicher und präziser Studienberater der FHDW.\n"
         "Nutze den folgenden Kontext und den bisherigen Gesprächsverlauf, um die Nutzerfrage zu beantworten.\n"
         "Nutze ausschließlich den bereitgestellten Kontext und die Gesprächshistorie.\n"
         f"{typ_prompt.get(frage_typ)}\n\n"
         f"### Verlauf (Zusammenfassung) ###\n{verlaufszusammenfassung}\n\n"
         f"### Kontext ###\n{context}\n\n### Frage ###\n{frage}"
         f"### Kontext ###\n{kontext}\n\n### Frage ###\n{frage}"
     )
 
     return [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": frage}
     ]
 
 
 def summarize_session_history(history, max_tokens=800, model="gpt-4o-mini", api_key=None):
     if not history or not api_key:
         return ""