import asyncio

import httpx
import pytest
from inspect_ai.tool import ToolError

from eval.web_tools import ordered_url, web_fetch


def test_query_order_preserves_values():
    assert ordered_url('https://example.com/?year=1923&name=A%2BB&name=&a=x#part') == (
        'https://example.com/?a=x&name=A%2BB&name=&year=1923#part'
    )
    assert ordered_url('https://example.com/') == 'https://example.com/'


def test_fetch_sorts_initial_and_redirect_urls(monkeypatch):
    seen = []
    def handle(request):
        seen.append(str(request.url))
        if len(seen) == 1:
            return httpx.Response(302, headers={'location': '/?year=1924&name=Other'})
        return httpx.Response(400, text='Wrong query order: year before name')
    client = httpx.AsyncClient
    monkeypatch.setattr('eval.web_tools.httpx.AsyncClient',
                        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    result = asyncio.run(web_fetch()('https://example.com/?year=1923&name=Keynes'))
    assert seen == ['https://example.com/?name=Keynes&year=1923',
                    'https://example.com/?name=Other&year=1924']
    assert 'HTTP status: 400' in result
    assert 'Wrong query order' in result


def test_fetch_rejects_non_http():
    with pytest.raises(ToolError, match='HTTP'):
        asyncio.run(web_fetch()('file:///tmp/data'))


def test_exa_search_request_and_results(monkeypatch):
    import json
    from eval.web_tools import exa_search
    from inspect_ai.tool import ToolDef

    results = [{'title': 'Quote', 'url': 'https://example.com/', 'text': 'Café'}]
    def handle(request):
        assert request.method == 'POST'
        assert str(request.url) == 'https://api.exa.ai/search'
        assert request.headers['x-api-key'] == 'test-key'
        assert json.loads(request.content) == {
            'query': 'a quote', 'type': 'auto', 'numResults': 5,
            'contents': {'text': {'maxCharacters': 4000}},
        }
        return httpx.Response(200, json={'results': results, 'requestId': 'ignored'})
    client = httpx.AsyncClient
    monkeypatch.setenv('EXA_API_KEY', 'test-key')
    monkeypatch.setattr('eval.web_tools.httpx.AsyncClient',
                        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    search = exa_search()
    assert ToolDef(search).name == 'web_search'
    assert json.loads(asyncio.run(search('a quote'))) == {'results': results}


@pytest.mark.parametrize('status,payload,expected', [
    (200, {'results': []}, None),
    (200, {'unexpected': []}, 'Expected an Exa results list'),
    (401, {}, 'HTTP 401'),
    (429, {}, 'HTTP 429'),
    (500, {}, 'HTTP 500'),
])
def test_exa_empty_and_error_responses(monkeypatch, status, payload, expected):
    from eval.web_tools import exa_search
    client = httpx.AsyncClient
    monkeypatch.setenv('EXA_API_KEY', 'test-key')
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=payload))
    monkeypatch.setattr('eval.web_tools.httpx.AsyncClient',
                        lambda **kwargs: client(transport=transport, **kwargs))
    if expected:
        with pytest.raises(ToolError, match=expected):
            asyncio.run(exa_search()('a quote'))
    else:
        assert asyncio.run(exa_search()('a quote')) == '{"results": []}'


def test_exa_requires_key(monkeypatch):
    from eval.web_tools import exa_search
    monkeypatch.delenv('EXA_API_KEY', raising=False)
    with pytest.raises(ToolError, match='EXA_API_KEY'):
        asyncio.run(exa_search()('a quote'))


@pytest.mark.parametrize('layers', [1, 2, 3])
def test_sorts_encoded_jina_target(layers):
    from urllib.parse import quote, unquote
    target = 'https://example.com/quotes/?year=2006&name=Winston+Churchill'
    encoded = target
    for _ in range(layers):
        encoded = quote(encoded, safe='')
    result = ordered_url('https://r.jina.ai/' + encoded)
    decoded = result.removeprefix('https://r.jina.ai/')
    for _ in range(layers):
        decoded = unquote(decoded)
    assert decoded == 'https://example.com/quotes/?name=Winston+Churchill&year=2006'
    assert ordered_url(result) == result


def test_sorts_partially_encoded_path_and_nested_query_target():
    from urllib.parse import quote, unquote, parse_qs, urlsplit
    target = 'https://example.com/quotes/?year=2006&name=A%26B%2BC'
    partial = 'https://r.jina.ai/' + quote(target, safe='/:')
    result = ordered_url(partial)
    assert unquote(result.removeprefix('https://r.jina.ai/')) == (
        'https://example.com/quotes/?name=A%26B%2BC&year=2006'
    )
    wrapped = 'https://proxy.example/?url=' + quote(partial, safe='')
    assert parse_qs(urlsplit(ordered_url(wrapped)).query)['url'] == [result]


def test_fetch_orders_encoded_target_after_redirect(monkeypatch):
    from urllib.parse import quote
    target = 'https://example.com/quotes/?year=2006&name=Winston+Churchill'
    expected = 'https://r.jina.ai/' + quote(
        'https://example.com/quotes/?name=Winston+Churchill&year=2006', safe='')
    seen = []
    def handle(request):
        seen.append(str(request.url))
        if len(seen) == 1:
            return httpx.Response(302, headers={
                'location': 'https://r.jina.ai/' + quote(target, safe='')})
        return httpx.Response(200, text='Target query has been ordered')
    client = httpx.AsyncClient
    monkeypatch.setattr('eval.web_tools.httpx.AsyncClient',
                        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    asyncio.run(web_fetch()('https://example.com/'))
    assert seen == ['https://example.com/', expected]
