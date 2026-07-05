# Yandex Maps Relevance Agent App

Minimal local web application for the LLM/search agent.

Run:

```bash
python app/agent_web_app.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

The app evaluates one pair:

- user query;
- organization name;
- rubric;
- address;
- services/prices summary;
- reviews/description summary.

The agent:

1. makes a local relevance assessment;
2. decides whether external search is needed;
3. optionally calls Serper or Tavily;
4. returns one of the discrete labels: `0.0`, `0.1`, `1.0`;
5. returns evidence and explanation as JSON.

Configuration can be entered in the UI or provided through environment variables:

```bash
copy .env.example .env
notepad .env
```

For RouterAI, keep these values in `.env`:

```bash
ROUTERAI_API_KEY=...
ROUTERAI_BASE_URL=https://routerai.ru/api/v1
AGENT_MODEL=deepseek/deepseek-v4-pro
SEARCH_PROVIDER=none
```

Optional search:

```bash
SEARCH_PROVIDER=serper
SERPER_API_KEY=...
```

or:

```bash
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=...
```
