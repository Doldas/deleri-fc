import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from strategy import ASSIGNMENTS, _discard_match, decide, start_match


def _boot_marker() -> str:
    from src.evolution import genome_hash
    from src.runtime import load_evolved_genome

    genome = load_evolved_genome()
    return genome_hash(genome)[:16] if genome else "none"


print(f"MVTEAM-BOOT artifact-hash={_boot_marker()}", flush=True)


TEAM = json.loads(Path(__file__).with_name("team.json").read_text(encoding="utf-8"))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/health":
            self.send_json(200, {"status": "ready"})
        elif self.path == "/v1/team":
            self.send_json(200, TEAM)
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.read_body())
            if self.path == "/v1/matches/end":
                _discard_match(body.get("gameId", ""))
                self.send_response(204)
                self.end_headers()
                return
            if self.path == "/v1/matches/start":
                ASSIGNMENTS.pop(body.get("gameId", ""), None)
                start_match(body)
                self.send_json(200, {"ready": True})
            elif self.path == "/v1/decide":
                self.send_json(200, decide(body))
            else:
                self.send_json(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError) as error:
            self.send_json(400, {"error": str(error)})

    def read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks, total = [], 0
            while True:
                size = int(self.rfile.readline().strip().split(b";", 1)[0], 16)
                if size == 0:
                    self.rfile.readline()
                    break
                total += size
                if total > 128 * 1024:
                    raise ValueError("request too large")
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length", "0"))
        if length > 128 * 1024:
            raise ValueError("request too large")
        return self.rfile.read(length)

    def send_json(self, status, value):
        payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
