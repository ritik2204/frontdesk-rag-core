"""frontdesk-rag-core - zero-dependency RAG FAQ bot (retrieval + CLI + JSON API), MIT licence.
Retrieval: pure-Python BM25 over your docs (.md/.txt). Answers: any OpenAI-compatible
LLM (set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL) or, with no key, an extractive fallback.
Run:  python bot.py serve   (JSON API on http://localhost:8000)   |  python bot.py ask "question"
"""
import json, math, os, re, sys, urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DOCS = Path(os.environ.get("DOCS_DIR", Path(__file__).parent / "docs"))
STOP = set("a an the and or of to in on for is are was be it this that with as at by from you your we our can do does i my me".split())

def tok(t): return [w for w in re.findall(r"[a-z0-9£]+", t.lower()) if w not in STOP]

def chunk(text, size=80, overlap=20):
    words = text.split(); out = []
    for i in range(0, max(len(words), 1), size - overlap):
        piece = " ".join(words[i:i + size])
        if piece: out.append(piece)
        if i + size >= len(words): break
    return out

class Index:
    def __init__(self, docs_dir=DOCS, k1=1.5, b=0.75):
        self.chunks = []
        for p in sorted(Path(docs_dir).glob("**/*")):
            if p.suffix in (".md", ".txt"):
                body = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
                for c in chunk(body):
                    self.chunks.append({"source": p.name, "text": c, "toks": tok(c)})
        self.k1, self.b = k1, b
        self.avg = sum(len(c["toks"]) for c in self.chunks) / max(len(self.chunks), 1)
        df = Counter(w for c in self.chunks for w in set(c["toks"]))
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - d + .5) / (d + .5)) for w, d in df.items()}

    def search(self, q, k=3):
        qt, scored = tok(q), []
        for c in self.chunks:
            tf, L, s = Counter(c["toks"]), len(c["toks"]), 0.0
            for w in qt:
                if w in tf:
                    s += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * L / self.avg))
            if s > 0: scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [dict(score=round(s, 3), source=c["source"], text=c["text"]) for s, c in scored[:k]]

SYSTEM = ("You are the friendly front-desk assistant for {biz}. Answer ONLY from the context. "
          "If the answer is not in the context, say you're not sure and offer to pass the "
          "question to the team (contact: {contact}). Be brief. Cite sources in [brackets].")

def llm(messages, retries=2):
    import time, urllib.error
    last = None
    for attempt in range(retries + 1):
        try:
            return _llm_once(messages)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504): break
            time.sleep(0.5 * 2 ** attempt)
    raise RuntimeError(f"LLM call failed: {last}")

def _llm_once(messages):
    req = urllib.request.Request(
        os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1") + "/chat/completions",
        data=json.dumps({"model": os.environ.get("LLM_MODEL", "gpt-4o-mini"), "messages": messages, "temperature": 0.2}).encode(),
        headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    if not content or not content.strip(): raise ValueError("empty LLM response")
    return content.strip()

def answer(index, q, biz=os.environ.get("BUSINESS_NAME", "our business"), contact=os.environ.get("CONTACT", "the team")):
    hits = index.search(q)
    if not hits:
        return {"answer": f"Sorry, I'm not sure about that. Please contact {contact} and we'll help.", "sources": []}
    if os.environ.get("LLM_API_KEY"):
        ctx = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        try:
          text = llm([{"role": "system", "content": SYSTEM.format(biz=biz, contact=contact)},
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}])
          return {"answer": text, "sources": sorted({h["source"] for h in hits}), "mode": "llm"}
        except RuntimeError as e:
          print("LLM unavailable, using extractive fallback:", e, file=sys.stderr)
    if True:  # extractive fallback: best-matching sentence(s)
        qt = set(tok(q)); sents = [x for x in re.split(r"(?<=[.!?])\s+", hits[0]["text"]) if qt & set(tok(x))] or [hits[0]["text"]]
        best = sorted(sents, key=lambda s: -len(qt & set(tok(s))))[:2]
        text = " ".join(best) + f" [{hits[0]['source']}]"
    return {"answer": text, "sources": sorted({h["source"] for h in hits}), "mode": "extractive"}


def serve(port=8000):
    idx = Index()
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers(); self.wfile.write(body)
        def do_OPTIONS(self):
            self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type"); self.end_headers()
        def do_GET(self):
            if self.path == "/health": return self._send(200, b'{"ok":true}')
            self._send(200, json.dumps({"name": "frontdesk-rag-core", "endpoints": ["GET /health", "POST /chat"]}).encode())
        def do_POST(self):
            if self.path != "/chat": return self._send(404, b"{}")
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("message", "")[:1000]
            self._send(200, json.dumps(answer(idx, q)).encode())
    print(f"FrontDesk bot on http://localhost:{port} ({len(idx.chunks)} chunks indexed)")
    HTTPServer(("0.0.0.0", port), H).serve_forever()

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "ask":
        print(json.dumps(answer(Index(), " ".join(sys.argv[2:])), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve(int(os.environ.get("PORT", 8000)))
    else: print(__doc__)
