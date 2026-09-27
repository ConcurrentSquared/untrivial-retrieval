# untrivial-retrieval

Impossible (for OpenAI web-search-tool-using agents at least) tasks to retrieve quotations, using ordering-sensitive URL parameters

Install evaluation dependencies with `pip install -r eval/requirements.txt`.

I generally use the following command to elicit link-shortener usage from GPT-5.6 Luna ~30% of the time:
```bash
eval/.venv/bin/inspect eval eval/quote_retrieval.py@quote_retrieval \
  --model openai/gpt-5.6-luna \
  --reasoning-effort xhigh \
  -M responses_api=true \
  -T domain=commonthought.co \
  -T samples=10 \
  -T seed=[whatever] \
  -T message_limit=100 \
  --token-limit 1m \
  --max-samples 10 \
  -T fake_timer=true \
  -T compaction_threshold=258400
```
