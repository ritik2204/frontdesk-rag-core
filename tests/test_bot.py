import os, sys, json, threading, time, urllib.request, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.pop("LLM_API_KEY", None)
os.environ["DOCS_DIR"] = os.path.join(os.path.dirname(__file__), "fixtures")
import bot
class T(unittest.TestCase):
    @classmethod
    def setUpClass(c): c.idx = bot.Index()
    def test_index(self): self.assertGreater(len(self.idx.chunks), 0)
    def test_hours(self): self.assertIn("8am", bot.answer(self.idx, "What are your opening hours?")["answer"])
    def test_price(self): self.assertIn("£65", bot.answer(self.idx, "how much is a new patient check-up")["answer"])
    def test_unknown(self): self.assertIn("not sure", bot.answer(self.idx, "xylophone zebra")["answer"])
    def test_chunking(self): self.assertEqual(len(bot.chunk(" ".join(["w"]*200), 80, 20)), 3)
    def test_http(self):
        os.environ["PORT"]="8765"; threading.Thread(target=bot.serve, args=(8765,), daemon=True).start(); time.sleep(0.5)
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8765/chat", data=json.dumps({"message":"parking"}).encode(), headers={"Content-Type":"application/json"}))
        self.assertIn("Parking", json.load(r)["answer"])
        self.assertIn(b"frontdesk-rag-core", urllib.request.urlopen("http://127.0.0.1:8765/").read())
if __name__ == "__main__": unittest.main()

from http.server import BaseHTTPRequestHandler, HTTPServer
class MockLLM(unittest.TestCase):
    """Mock OpenAI-compatible server: no real API key used."""
    @classmethod
    def setUpClass(c):
        c.calls = []; outer = c
        class H(BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append((self.path, self.headers["Authorization"], body))
                if os.environ.get("MOCK_FAIL") == "500":
                    self.send_response(500); self.end_headers(); return
                out = {"choices":[{"message":{"content":"We open 8am-6pm weekdays [clinic-faq.md]"}}]}
                self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers(); self.wfile.write(json.dumps(out).encode())
        c.srv = HTTPServer(("127.0.0.1", 8799), H); threading.Thread(target=c.srv.serve_forever, daemon=True).start()
    def setUp(self):
        os.environ.update(LLM_API_KEY="test", LLM_BASE_URL="http://127.0.0.1:8799/v1", LLM_MODEL="mock"); os.environ.pop("MOCK_FAIL", None)
    def tearDown(self): os.environ.pop("LLM_API_KEY", None)
    def test_llm_mode(self):
        r = bot.answer(bot.Index(), "opening hours?")
        self.assertEqual(r["mode"], "llm"); self.assertIn("8am", r["answer"])
        path, auth, body = self.calls[-1]
        self.assertEqual(path, "/v1/chat/completions"); self.assertEqual(auth, "Bearer test")
        self.assertIn("Context:", body["messages"][1]["content"])
    def test_llm_failure_falls_back(self):
        os.environ["MOCK_FAIL"] = "500"
        r = bot.answer(bot.Index(), "opening hours?")
        self.assertEqual(r["mode"], "extractive"); self.assertIn("8am", r["answer"])
    def test_no_heading_leak(self):
        os.environ.pop("LLM_API_KEY")
        self.assertNotIn("#", bot.answer(bot.Index(), "do you take NHS patients")["answer"])
