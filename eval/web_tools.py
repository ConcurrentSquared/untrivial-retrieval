"""Client-side Exa search and fetching with alphabetized query parameters."""
import json
import os
import re
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from inspect_ai.tool import Tool, ToolError, tool


@tool(name='web_search')
def exa_search() -> Tool:
    async def execute(query: str) -> str:
        """Search the web for relevant pages using Exa.

        Args:
            query: Search query.
        """
        api_key = os.environ.get('EXA_API_KEY')
        if not api_key:
            raise ToolError('Set EXA_API_KEY to use Exa search')
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    'https://api.exa.ai/search',
                    headers={'x-api-key': api_key},
                    json={'query': query, 'type': 'auto', 'numResults': 5,
                          'contents': {'text': {'maxCharacters': 4000}}},
                )
                response.raise_for_status()
                data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get('results'), list):
                raise ValueError('Expected an Exa results list')
            return json.dumps({'results': data['results']}, ensure_ascii=False)
        except httpx.HTTPStatusError as exc:
            raise ToolError(f'Exa search failed (HTTP {exc.response.status_code})') from exc
        except (ValueError, httpx.HTTPError) as exc:
            raise ToolError(f'Exa search failed: {exc}') from exc
    return execute


def _embedded_url(value: str, depth: int) -> str:
    """Normalize a URL carried as a path suffix or query value."""
    if depth > 20:
        raise ValueError('Embedded URL nesting is too deep')
    # Decode only URL envelopes, not arbitrary parameter values: decoding a
    # quote's literal %26 into a separator would change its meaning.
    encoded_scheme = re.match(r'https?%(?:25)*3[aA]', value)
    encoded_query = (value.startswith(('http://', 'https://'))
                     and '?' not in value and re.search(r'%(?:25)*3[fF]', value))
    if encoded_scheme or encoded_query:
        decoded = unquote(value)
        normalized = _embedded_url(decoded, depth + 1)
        if normalized == decoded:
            return value
        return quote(normalized, safe='' if encoded_scheme else '/:')
    if value.startswith(('http://', 'https://')):
        return ordered_url(value, _depth=depth + 1)
    return value


def ordered_url(url: str, *, _depth: int = 0) -> str:
    if _depth > 20:
        raise ValueError('Embedded URL nesting is too deep')
    parts = urlsplit(url)
    if parts.scheme not in {'http', 'https'} or not parts.hostname:
        raise ValueError('Use an absolute HTTP or HTTPS URL')
    # Stable sorting preserves repeated parameters and their value order.
    pairs = sorted(parse_qsl(parts.query, keep_blank_values=True), key=lambda p: p[0])
    pairs = [(key, _embedded_url(value, _depth + 1)) for key, value in pairs]
    path = parts.path
    embedded = re.search(r'https?(?:://|%(?:25)*3[aA])', path)
    if embedded:
        path = path[:embedded.start()] + _embedded_url(path[embedded.start():], _depth + 1)
    return urlunsplit(parts._replace(path=path, query=urlencode(pairs)))


@tool
def web_fetch() -> Tool:
    async def execute(url: str) -> str:
        """Open a web page and return its contents.

        Args:
            url: Absolute URL of the page to open.
        """
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                for _ in range(11):
                    url = ordered_url(url)
                    response = await client.get(url)
                    if response.has_redirect_location:
                        url = urljoin(str(response.url), response.headers['location'])
                        continue
                    # Include error bodies: the quote server explains bad query order there.
                    return f'URL: {response.url}\nHTTP status: {response.status_code}\n\n{response.text}'
                raise ToolError('Too many redirects')
        except (ValueError, httpx.HTTPError) as exc:
            raise ToolError(str(exc)) from exc
    return execute
