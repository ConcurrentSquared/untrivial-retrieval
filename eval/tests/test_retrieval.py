import asyncio
import copy

import pytest
from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef

from eval.quote_retrieval import domain_name, parse_answer, quote_retrieval, quote_retrieval_suite

@pytest.fixture(autouse=True)
def compaction_without_api(monkeypatch):
    # Solver tests inject generation; avoid resolving a real model/credentials.
    class Compact:
        async def compact_input(self, messages):
            return list(messages), None

        async def record_output(self, messages, output):
            pass

    monkeypatch.setattr('eval.quote_retrieval.compaction', lambda *args, **kwargs: Compact())


DOMAIN = 'quotes.example.com'
ID = 'A' * 43 + '='  # Deliberately not a correct quote hash.
GOOD = f'ANSWER: {ID}\nCITATION: {DOMAIN}'


@pytest.mark.parametrize('response', [
    'The ID does not exist.', 'ANSWER: UNKNOWN\nCITATION: quotes.example.com',
    'ANSWER: not found\nCITATION: quotes.example.com',
    GOOD.replace(DOMAIN, 'other.example.com'),
    GOOD.replace(DOMAIN, DOMAIN + '.evil.com'),
    GOOD.replace(DOMAIN, 'sub.' + DOMAIN),
    GOOD.replace(DOMAIN, 'https://' + DOMAIN),
    GOOD + '\nExtra explanation',
])
def test_retry(response):
    assert parse_answer(response, DOMAIN)[0] is None


def test_format_not_correctness():
    assert parse_answer(GOOD, DOMAIN)[0] == ID
    assert parse_answer(GOOD.upper(), DOMAIN)[0] == ID
    with pytest.raises(ValueError):
        domain_name('https://' + DOMAIN)


def test_dataset_and_suite():
    first = quote_retrieval(DOMAIN, samples=3)
    second = quote_retrieval(DOMAIN, samples=3)
    assert [s.metadata for s in first.dataset] == [s.metadata for s in second.dataset]
    assert len(first.dataset) == 3
    for sample in first.dataset:
        assert len({q['text'] for q in sample.metadata['quotes']}) == 5
        assert all(set(q) == {'text', 'author', 'year'} for q in sample.metadata['quotes'])
        assert not sample.target
    assert not first.scorer
    suite = quote_retrieval_suite(DOMAIN + ',other.example.com', samples=1).initial_tasks()
    assert len({t.name for t in suite}) == 2


def state_for(task, index=0, model='openai/gpt-5'):
    sample = task.dataset[index]
    return TaskState(model=ModelName(model), sample_id=sample.id, epoch=1,
                     input=sample.input, messages=[ChatMessageUser(content=sample.input)],
                     metadata=copy.deepcopy(sample.metadata))


@pytest.mark.parametrize("fake_timer", [False, True])
def test_concurrent_rounds_retry_and_independent_state(fake_timer):
    task = quote_retrieval(DOMAIN, samples=2, fake_timer=fake_timer)

    async def run(index):
        state = state_for(task, index)
        responses = iter(['Unknown', GOOD.replace(DOMAIN, 'wrong.example.com')] + [GOOD] * 5)
        async def generate(state, **kwargs):
            await asyncio.sleep(0)
            state.output = ModelOutput.from_content('test', next(responses))
            state.messages.append(state.output.message)
            return state
        return await task.solver(state, generate)

    async def together():
        return await asyncio.gather(run(0), run(1))
    states = asyncio.run(together())
    for state in states:
        assert state.metadata['rounds_completed'] == 5
        assert [len(r['attempts']) for r in state.metadata['rounds']] == [3, 1, 1, 1, 1]
        retries = [m.content for m in state.messages
                   if isinstance(m, ChatMessageUser) and 'additional' in m.content]
        assert len(retries) == (2 if fake_timer else 0)
        if fake_timer:
            assert 'additional 156 seconds; you now have 159 seconds.' in retries[0]
            assert 'additional 155 seconds; you now have 158 seconds.' in retries[1]
        round_prompts = [m.content for m in state.messages
                         if isinstance(m, ChatMessageUser) and m.content.startswith('Round ')]
        assert len(round_prompts) == 5
        assert all(('You have 160 seconds.' in p) == fake_timer for p in round_prompts)
        assert len(state.tools) == 1
        assert ToolDef(state.tools[0]).name == 'web_search'
    assert states[0].metadata['rounds'] is not states[1].metadata['rounds']


def test_limit_does_not_advance_failed_round():
    task = quote_retrieval(DOMAIN, samples=1)
    async def generate(state, **kwargs):
        state.output = ModelOutput.from_content('test', 'Unknown')
        state.completed = True
        return state
    state = asyncio.run(task.solver(state_for(task), generate))
    assert state.metadata['rounds_completed'] == 0
    assert len(state.metadata['rounds']) == 1


@pytest.mark.parametrize('model,provider,names', [
    ('openrouter/anthropic/claude-sonnet-4', 'auto', ['web_search', 'web_fetch']),
    ('openai/gpt-5', 'exa', ['web_search', 'web_fetch']),
    ('openai/gpt-5', 'auto', ['web_search']),
])
def test_search_selection(model, provider, names):
    task = quote_retrieval(DOMAIN, samples=1, search_provider=provider)
    state = state_for(task, model=model)
    async def generate(state, **kwargs):
        state.completed = True
        state.output = ModelOutput.from_content('test', GOOD)
        return state
    state = asyncio.run(task.solver(state, generate))
    assert [ToolDef(t).name for t in state.tools] == names


def test_invalid_search_provider():
    with pytest.raises(ValueError, match='search_provider'):
        quote_retrieval(DOMAIN, search_provider='invalid')


def test_native_search_requires_openai():
    task = quote_retrieval(DOMAIN, search_provider='openai')
    state = state_for(task, model='openrouter/anthropic/claude-sonnet-4')
    with pytest.raises(ValueError, match='requires an openai'):
        asyncio.run(task.solver(state, None))


def test_inspect_file_loader_without_repo_on_pythonpath(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    task_file = Path(__file__).resolve().parents[1] / 'quote_retrieval.py'
    script = '''
import sys
from pathlib import Path
from inspect_ai._eval.loader import load_file_tasks, create_file_tasks
path = Path(sys.argv[1])
load_file_tasks(path)
tasks = create_file_tasks(path, ['quote_retrieval'],
                          {'domain': 'quotes.example.com', 'samples': 1,
                           'search_provider': 'exa'})
assert len(tasks) == 1
assert len(tasks[0].dataset) == 1
'''
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-I', '-c', script, str(task_file)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('provider,engine', [
    ('openrouter-native', 'native'), ('openrouter-exa', 'exa'),
])
def test_openrouter_server_search(provider, engine, monkeypatch):
    monkeypatch.delenv('EXA_API_KEY', raising=False)
    task = quote_retrieval(DOMAIN, samples=1, search_provider=provider)
    assert task.config.extra_body == {'plugins': [{'id': 'web', 'engine': engine}]}
    state = state_for(task, model='openrouter/test/model')
    async def generate(state, **kwargs):
        assert [ToolDef(t).name for t in state.tools] == ['web_fetch']
        state.output = ModelOutput.from_content('test', GOOD)
        return state
    result = asyncio.run(task.solver(state, generate))
    assert result.metadata['rounds_completed'] == 5
    with pytest.raises(ValueError, match=r'requires an openrouter/\* model'):
        asyncio.run(task.solver(state_for(task), generate))
    suite = quote_retrieval_suite(DOMAIN + ',other.example.com', samples=1,
                                 search_provider=provider).initial_tasks()
    assert all(t.config.extra_body == task.config.extra_body for t in suite)


@pytest.mark.parametrize('provider', ['auto', 'openai', 'exa'])
def test_client_search_has_no_openrouter_plugin(provider):
    assert quote_retrieval(DOMAIN, samples=1, search_provider=provider).config.extra_body is None


@pytest.mark.parametrize('engine', ['native', 'exa'])
def test_openrouter_plugin_on_wire(engine):
    import json
    import httpx
    from inspect_ai.model import GenerateConfig
    from inspect_ai.tool import ToolInfo
    from eval.web_tools import web_fetch
    from inspect_ai.model._providers.openrouter import OpenRouterAPI

    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': 'test/model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': GOOD}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            api = OpenRouterAPI('test/model', api_key='test-key', http_client=client,
                                provider={'only': ['test-provider'], 'allow_fallbacks': False},
                                stream=False)
            task = quote_retrieval(DOMAIN, samples=1, search_provider='openrouter-' + engine)
            config = task.config.merge(GenerateConfig(reasoning_effort='xhigh'))
            fetch = ToolDef(web_fetch())
            await api.generate([ChatMessageUser(content='Find the quote')],
                               [ToolInfo(name=fetch.name, description=fetch.description,
                                         parameters=fetch.parameters)],
                               'auto', config)
    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0]['plugins'] == [{'id': 'web', 'engine': engine}]
    assert requests[0]['provider'] == {'only': ['test-provider'], 'allow_fallbacks': False}
    assert requests[0]['reasoning']['effort'] == 'xhigh'
    assert [t['function']['name'] for t in requests[0]['tools']] == ['web_fetch']


@pytest.mark.parametrize('model,strategy_name', [
    ('openai/gpt-5', 'CompactionNative'),
    ('anthropic/claude-sonnet-4-5', 'CompactionNative'),
    ('openrouter/anthropic/claude-sonnet-4', 'CompactionSummary'),
    ('google/gemini-2.5-pro', 'CompactionSummary'),
])
@pytest.mark.parametrize('threshold', [0.9, 0.8, 256000])
def test_compaction_provider_and_sample_isolation(model, strategy_name, threshold, monkeypatch):
    import importlib
    module = importlib.import_module('eval.quote_retrieval')
    original = module.compaction
    handlers = []

    def create(strategy, **kwargs):
        assert type(strategy).__name__ == strategy_name
        assert strategy.threshold == threshold
        assert type(strategy.threshold) is type(threshold)
        assert kwargs['prefix'][0].role == 'system'
        assert kwargs['prefix'][1].role == 'user'
        assert kwargs['tools']
        handler = original(strategy, **kwargs)
        handlers.append(handler)
        return handler

    monkeypatch.setattr(module, 'compaction', create)
    task = quote_retrieval(DOMAIN, samples=2, compaction_threshold=threshold)

    async def generate(state, **kwargs):
        state.output = ModelOutput.from_content('test', GOOD)
        state.messages.append(state.output.message)
        return state

    async def run():
        return await asyncio.gather(*[
            task.solver(state_for(task, i, model), generate) for i in range(2)
        ])

    states = asyncio.run(run())
    assert all(state.metadata['rounds_completed'] == 5 for state in states)
    assert len(handlers) == 2 and handlers[0] is not handlers[1]


def test_compaction_between_tool_steps_preserves_history(monkeypatch):
    from inspect_ai.model import ChatMessageTool
    from inspect_ai.tool import ToolCall
    from eval.quote_retrieval import generate_compacted

    state = state_for(quote_retrieval(DOMAIN, samples=1))
    original = state.messages[0]
    summary = ChatMessageUser(content='Summary of the current retrieval round')
    inputs, recorded = [], []

    class Compact:
        async def compact_input(self, messages):
            inputs.append(list(messages))
            return [summary], None

        async def record_output(self, messages, output):
            recorded.append(list(messages))

    async def generate(state, tool_calls):
        assert tool_calls == 'none'
        assert list(state.messages) == [summary]
        state.output = ModelOutput.from_content('test', GOOD if len(inputs) == 2 else '')
        if len(inputs) == 1:
            state.output.message.tool_calls = [ToolCall(id='call1', function='web_fetch', arguments={})]
        state.messages.append(state.output.message)
        return state

    async def execute_tools(messages, tools):
        assert messages[0] == original
        return [ChatMessageTool(content='page content', tool_call_id='call1')], None

    monkeypatch.setattr('eval.quote_retrieval.execute_tools', execute_tools)
    state = asyncio.run(generate_compacted(state, generate, Compact()))
    assert len(inputs) == 2
    assert len(inputs[1]) == 3
    assert state.messages[0] == original
    assert len(state.messages) == 4
    assert all(messages == [summary] for messages in recorded)
    assert state.output.completion == GOOD


def test_compaction_restores_history_on_generation_error():
    from eval.quote_retrieval import generate_compacted
    state = state_for(quote_retrieval(DOMAIN, samples=1))
    history = list(state.messages)

    class Compact:
        async def compact_input(self, messages):
            return [ChatMessageUser(content='summary')], None

    async def generate(state, **kwargs):
        raise RuntimeError('provider failed')

    with pytest.raises(RuntimeError, match='provider failed'):
        asyncio.run(generate_compacted(state, generate, Compact()))
    assert list(state.messages) == history


def test_real_inspect_summary_compaction(tmp_path, monkeypatch):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import CompactionSummary, compaction, get_model

    def create(strategy, **kwargs):
        return compaction(CompactionSummary(threshold=5000, memory=False), **kwargs)

    monkeypatch.setattr('eval.quote_retrieval.compaction', create)
    monkeypatch.setenv('INSPECT_TRACE_FILE', str(tmp_path / 'trace.log'))
    monkeypatch.setattr('inspect_ai._util.appdirs.user_data_path', lambda *args: tmp_path / 'data')
    monkeypatch.setattr('inspect_ai._util.appdirs.user_cache_path', lambda *args: tmp_path / 'cache')
    model = get_model('mockllm/model', custom_outputs=lambda *args: ModelOutput.from_content('mockllm', GOOD))
    async def count_tokens(messages, tools=None):
        return sum(len(message.text) for message in messages)

    async def count_text_tokens(text):
        return len(text)

    # Deterministic token counts avoid downloading tokenizer assets in tests.
    monkeypatch.setattr(model.api, 'count_tokens', count_tokens)
    monkeypatch.setattr(model.api, 'count_text_tokens', count_text_tokens)
    logs = inspect_eval(quote_retrieval(DOMAIN, samples=1), model=model,
                        log_dir=str(tmp_path), display='none')
    assert logs[0].status == 'success', logs[0].error
    sample = logs[0].samples[0]
    assert sample.metadata['rounds_completed'] == 5
    assert any(event.event == 'compaction' for event in sample.events)
    assert sum(m.role == 'assistant' for m in sample.messages) == 5


@pytest.mark.parametrize('threshold', [0, -1, 1.5, 256000.0, True, '256k'])
def test_invalid_compaction_threshold(threshold):
    with pytest.raises(ValueError, match='compaction_threshold'):
        quote_retrieval(DOMAIN, compaction_threshold=threshold)


def test_suite_compaction_threshold():
    from inspect_ai._util.registry import registry_params
    tasks = quote_retrieval_suite(DOMAIN + ',other.example.com', samples=1,
                                  compaction_threshold=256000).initial_tasks()
    assert all(registry_params(task.solver)['compaction_threshold'] == 256000 for task in tasks)
