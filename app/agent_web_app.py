import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_runner import DEFAULT_AGENT_MODEL, DEFAULT_BASE_URL, load_env_file, run_case  # noqa: E402


load_env_file(ROOT / ".env")


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yandex Maps Relevance Agent</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #172033;
      --muted: #5d6678;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      --danger: #b42318;
      --ok: #067647;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 18px 24px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }
    main {
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: minmax(360px, 0.92fr) minmax(420px, 1.08fr);
      gap: 20px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-width: 0;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 15px;
      font-weight: 700;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      font-size: 13px;
      font-weight: 650;
      color: var(--muted);
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      color: var(--text);
      background: #fff;
      outline: none;
    }
    textarea { min-height: 76px; resize: vertical; }
    textarea.tall { min-height: 118px; }
    input:focus, textarea:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .actions {
      margin-top: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      background: var(--accent);
      color: #fff;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary {
      background: #eef2ff;
      color: #1e3a8a;
    }
    button.secondary:hover { background: #dbe4ff; }
    .status {
      min-height: 22px;
      font-size: 13px;
      color: var(--muted);
    }
    .result-top {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }
    .metric .name {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .metric .value {
      font-size: 22px;
      font-weight: 800;
      min-height: 30px;
      overflow-wrap: anywhere;
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--line);
      background: #fbfcfe;
      border-radius: 8px;
      padding: 12px;
      min-height: 160px;
      margin: 0;
      font-size: 13px;
    }
    .error { color: var(--danger); }
    .ok { color: var(--ok); }
    .muted { color: var(--muted); }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; padding: 14px; }
      .grid-2, .result-top { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Yandex Maps Relevance Agent</h1>
  </header>
  <main>
    <section>
      <h2>Карточка организации</h2>
      <form id="agent-form">
        <label for="query">Запрос пользователя</label>
        <input id="query" name="Text" value="кафе с верандой в москве недорого" required>

        <label for="name">Название</label>
        <input id="name" name="name" value="Гранатовый сад">

        <label for="rubric">Рубрика</label>
        <input id="rubric" name="normalized_main_rubric_name_ru" value="Ресторан">

        <label for="address">Адрес</label>
        <input id="address" name="address" value="Москва">

        <label for="prices">Товары и услуги</label>
        <textarea id="prices" name="prices_summarized"></textarea>

        <label for="reviews">Отзывы и описание</label>
        <textarea id="reviews" class="tall" name="reviews_summarized"></textarea>

        <h2 style="margin-top:18px">Настройки агента</h2>
        <div class="grid-2">
          <div>
            <label for="model">LLM model</label>
            <input id="model" name="model" value="deepseek/deepseek-v4-pro">
          </div>
          <div>
            <label for="base_url">Base URL</label>
            <input id="base_url" name="base_url" value="https://routerai.ru/api/v1">
          </div>
        </div>

        <label for="api_key">API key</label>
        <input id="api_key" name="api_key" type="password" placeholder="ROUTERAI_API_KEY или ключ совместимого gateway">

        <div class="grid-2">
          <div>
            <label for="search_provider">Поиск</label>
            <select id="search_provider" name="search_provider">
              <option value="none">none</option>
              <option value="serper">serper</option>
              <option value="tavily">tavily</option>
            </select>
          </div>
          <div>
            <label for="search_key">Search API key</label>
            <input id="search_key" name="search_key" type="password" placeholder="SERPER_API_KEY или TAVILY_API_KEY">
          </div>
        </div>

        <div class="actions">
          <button type="submit">Оценить</button>
          <button type="button" class="secondary" id="fill-example">Пример</button>
          <span class="status" id="status"></span>
        </div>
      </form>
    </section>

    <section>
      <h2>Решение агента</h2>
      <div class="result-top">
        <div class="metric">
          <div class="name">label</div>
          <div class="value" id="label">-</div>
        </div>
        <div class="metric">
          <div class="name">confidence</div>
          <div class="value" id="confidence">-</div>
        </div>
        <div class="metric">
          <div class="name">search</div>
          <div class="value" id="used_search">-</div>
        </div>
      </div>
      <h2>Объяснение</h2>
      <pre id="explanation" class="muted">После запуска здесь появятся evidence, план агента, поисковые запросы и финальное объяснение.</pre>
    </section>
  </main>

  <script>
    const form = document.getElementById("agent-form");
    const statusEl = document.getElementById("status");
    const labelEl = document.getElementById("label");
    const confidenceEl = document.getElementById("confidence");
    const usedSearchEl = document.getElementById("used_search");
    const explanationEl = document.getElementById("explanation");

    function formPayload() {
      const data = new FormData(form);
      const caseData = {};
      for (const key of ["Text", "name", "normalized_main_rubric_name_ru", "address", "prices_summarized", "reviews_summarized"]) {
        caseData[key] = data.get(key) || "";
      }
      return {
        case: caseData,
        model: data.get("model") || "deepseek/deepseek-v4-pro",
        base_url: data.get("base_url") || "https://routerai.ru/api/v1",
        api_key: data.get("api_key") || "",
        search_provider: data.get("search_provider") || "none",
        search_key: data.get("search_key") || ""
      };
    }

    function renderResult(result) {
      labelEl.textContent = result.label ?? "-";
      confidenceEl.textContent = result.final && result.final.confidence !== undefined ? result.final.confidence : "-";
      usedSearchEl.textContent = result.used_search ? "yes" : "no";
      const view = {
        final: result.final,
        plan: result.plan,
        search_results: result.search_results
      };
      explanationEl.className = "";
      explanationEl.textContent = JSON.stringify(view, null, 2);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusEl.textContent = "Агент думает...";
      statusEl.className = "status";
      explanationEl.className = "muted";
      try {
        const response = await fetch("/api/evaluate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(formPayload())
        });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.error || "Request failed");
        }
        renderResult(result);
        statusEl.textContent = "Готово";
        statusEl.className = "status ok";
      } catch (error) {
        labelEl.textContent = "-";
        confidenceEl.textContent = "-";
        usedSearchEl.textContent = "-";
        explanationEl.className = "error";
        explanationEl.textContent = String(error.message || error);
        statusEl.textContent = "Ошибка";
        statusEl.className = "status error";
      }
    });

    document.getElementById("fill-example").addEventListener("click", () => {
      document.getElementById("query").value = "итальянский ресторан метро третьяковская";
      document.getElementById("name").value = "Burger Heroes";
      document.getElementById("rubric").value = "Быстрое питание";
      document.getElementById("address").value = "Москва, улица Большая Ордынка, 19, стр. 1";
      document.getElementById("prices").value = "бургеры, картошка фри, куриные стрипсы, сырные палочки, чизкейк, кофе";
      document.getElementById("reviews").value = "Отзывы хвалят бургеры и быстрый сервис. Итальянская кухня не упоминается.";
    });
  </script>
</body>
</html>
"""


def make_args(payload: dict) -> SimpleNamespace:
    search_provider = payload.get("search_provider") or os.getenv("SEARCH_PROVIDER", "none")
    search_key = payload.get("search_key") or ""
    return SimpleNamespace(
        model=payload.get("model") or os.getenv("AGENT_MODEL", DEFAULT_AGENT_MODEL),
        base_url=(
            payload.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("ROUTERAI_BASE_URL", DEFAULT_BASE_URL)
        ),
        api_key=payload.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("ROUTERAI_API_KEY"),
        search_provider=search_provider,
        serper_api_key=search_key or os.getenv("SERPER_API_KEY"),
        tavily_api_key=search_key or os.getenv("TAVILY_API_KEY"),
        max_search_queries=2,
        max_search_results=4,
        fewshot_examples=[],
        fewshot_k=0,
        temperature=0.0,
    )


class AgentHandler(BaseHTTPRequestHandler):
    server_version = "RelevanceAgentHTTP/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self.send_json(200, {"ok": True})
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/evaluate":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            case = payload.get("case") or {}
            if not case.get("Text"):
                self.send_json(400, {"error": "Поле запроса обязательно."})
                return
            args = make_args(payload)
            result = run_case(case, args)
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(500, {"error": repr(exc)})

    def log_message(self, format: str, *args) -> None:
        print("%s - %s" % (self.address_string(), format % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local web UI for the relevance LLM agent.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AgentHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Agent app: {url}")
    print("Set ROUTERAI_API_KEY in .env or paste credentials in the form.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
