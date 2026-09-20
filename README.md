# untrivial-retrieval

Impossible (for OpenAI web-search-tool-using agents at least) tasks to retrieve quotations, using ordering-sensitive URL parameters

Install evaluation dependencies with `pip install -r eval/requirements.txt`.

Run native OpenAI search with `OPENAI_API_KEY` set:

```sh
inspect eval eval/quote_retrieval.py@quote_retrieval --model openai/gpt-5 -T domain=quotes.example.com
```

Run an OpenRouter model with `OPENROUTER_API_KEY` and `EXA_API_KEY` set:

```sh
inspect eval eval/quote_retrieval.py@quote_retrieval --model openrouter/anthropic/claude-sonnet-4 -T domain=quotes.example.com
```

`search_provider=auto` (the default) selects native search for OpenAI models and
Exa for other models, including OpenRouter. Pass `-T search_provider=exa` to use
Exa with OpenAI too, or `-T search_provider=openai` to require native OpenAI search.
The same option is available on `quote_retrieval_suite`.

Exa runs as a custom client-side `web_search` tool that calls the
[Exa search endpoint](https://exa.ai/docs/reference/search) directly using
`EXA_API_KEY`. It returns five search results with up to 4,000 characters of page
text each, alongside a `web_fetch` tool for opening URLs.
Fetching intentionally sorts query parameters alphabetically by name before every
HTTP request, including redirects. Duplicate parameters and blank values are
preserved. This reproduces the intended OpenAI fetch behavior for this experiment:
`?year=1923&name=Keynes` is requested as `?name=Keynes&year=1923`, so the server's
reverse-order requirement still causes a failure. HTTP error bodies are returned
to the model so it can read the server's explanation.

OpenRouter server-side search is a separate, opt-in feature:

| Task option | Search execution |
| --- | --- |
| `-T search_provider=exa` | Custom client-side Exa tool; requires `EXA_API_KEY` |
| `-T search_provider=openrouter-native` | OpenRouter web plugin with `engine: native` |
| `-T search_provider=openrouter-exa` | OpenRouter web plugin with `engine: exa` |

Both server-side modes require an `openrouter/*` model and `OPENROUTER_API_KEY`.
They do not use your `EXA_API_KEY`. Neither chooses the search engine automatically,
and the existing `auto` task setting still selects client-side Exa for OpenRouter.
The modes also work with `quote_retrieval_suite`.

For example, force OpenRouter's server-side Exa search:

```sh
eval/.venv/bin/inspect eval eval/quote_retrieval.py@quote_retrieval \
  --model openrouter/deepseek/deepseek-v4.1-flash \
  --reasoning-effort xhigh \
  -M 'provider={"only":["DeepSeek"],"allow_fallbacks":false}' \
  -T search_provider=openrouter-exa \
  -T domain=example.com \
  -T samples=10 \
  -T seed=0 \
  -T message_limit=100 \
  --max-samples 4 \
  -T fake_timer=true
```

For native search, replace `openrouter-exa` with `openrouter-native` and choose a
model/provider that supports native search. Unsupported native search can produce
an API error; this evaluation does not retry with Exa. These modes use Chat
Completions; leave `-M responses_api=true` out. Provider routing through
`-M provider=...` remains independent of the search engine.

The request uses `plugins: [{"id": "web", "engine": "native" | "exa"}]`, following
[OpenRouter's explicit engine selection](https://openrouter.ai/docs/guides/features/plugins/web-search#forcing-engine-selection).
It deliberately uses the web plugin: OpenRouter's newer `openrouter:web_search`
server tool documents an Exa fallback even with `engine: native`.
Do not override `plugins` via generation `extra_body` when using these task modes.

Both modes retain the local `web_fetch` tool with alphabetical query ordering.
Only requests made through that tool are normalized by this code; any search or
page fetching performed internally by OpenRouter follows OpenRouter's behavior.

Conversation compaction is enabled for both retrieval tasks using Inspect's
[compaction providers](https://inspect.aisi.org.uk/compaction.html).
Direct `openai/*` and `anthropic/*` models use `CompactionNative`; other providers
(including OpenRouter) use `CompactionSummary`. Compaction uses Inspect's default
90% context-window threshold and is checked between tool calls as well as rounds.
Set `-T compaction_threshold=256000` for an absolute threshold of 256,000 tokens,
or `-T compaction_threshold=0.8` for 80% of the context window. Both tasks accept
this option. Use the full integer token count, not a `256k` suffix.
Each sample keeps its own compaction state and retains its full conversation in
the evaluation log. Native compaction requires a model/API that supports it;
unsupported native models surface Inspect's error rather than silently switching
strategies.
