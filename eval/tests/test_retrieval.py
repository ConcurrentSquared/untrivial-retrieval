import asyncio
import copy

import pytest
from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef

from eval.quote_retrieval import domain_name, parse_answer, quote_retrieval, quote_retrieval_suite

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


def state_for(task, index=0):
    sample = task.dataset[index]
    return TaskState(model=ModelName('openai/gpt-5'), sample_id=sample.id, epoch=1,
                     input=sample.input, messages=[ChatMessageUser(content=sample.input)],
                     metadata=copy.deepcopy(sample.metadata))


def test_concurrent_rounds_retry_and_independent_state():
    task = quote_retrieval(DOMAIN, samples=2)

    async def run(index):
        state = state_for(task, index)
        responses = iter(['Unknown', GOOD.replace(DOMAIN, 'wrong.example.com')] + [GOOD] * 5)
        async def generate(state):
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
        assert len(state.tools) == 1
        assert ToolDef(state.tools[0]).name == 'web_search'
    assert states[0].metadata['rounds'] is not states[1].metadata['rounds']


def test_limit_does_not_advance_failed_round():
    task = quote_retrieval(DOMAIN, samples=1)
    async def generate(state):
        state.output = ModelOutput.from_content('test', 'Unknown')
        state.completed = True
        return state
    state = asyncio.run(task.solver(state_for(task), generate))
    assert state.metadata['rounds_completed'] == 0
    assert len(state.metadata['rounds']) == 1
