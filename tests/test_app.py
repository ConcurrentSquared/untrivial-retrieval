import json
import re

import pytest

from quote_server.app import create_app


@pytest.fixture
def client(tmp_path):
    rows = [{"author": author, "year": 1923, "text": f"Quote {author} {i}",
             "position": i, "source": "Book (1923)", "year_evidence": "Book (1923)",
             "url": "https://en.wikiquote.org/", "attribution": "Wikiquote contributors",
             "license": "CC BY-SA 4.0"}
            for author in ["John Maynard Keynes", "Other Author"] for i in range(8)]
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps({"quotes": rows}))
    return create_app(path).test_client()


def test_full_and_partial_filters(client):
    response = client.get('/?year=1923&name=John%20Maynard%20Keynes')
    assert response.status_code == 200
    assert response.content_type == "text/plain; charset=utf-8"
    assert response.text.count("Author:") == 8
    assert client.get('/?name=John%20Keynes').text.count("Author:") == 5
    assert client.get('/?year=1923').text.count("Author:") == 5
    assert client.get('/?year=1923&name=%22John%20Keynes%22').text.count("Author:") == 8
    assert client.get('/?year=1924&name=John%20Keynes').text == "No matching quotes.\n"


@pytest.mark.parametrize("query", [
    "name=John&year=1923", "name=a&name=b", "year=1923&year=1924", "foo=bar",
    "year=", "name=", "year=1923&name=", "year=-1", "year=0", "year=19.2",
    "year=12345", "year=１９２３", "name=%FF", "name=%ZZ", "name", "name=%22%22",
    "year=1923&name=John&extra=x", "year=1923&&name=John",
])
def test_reject_invalid_queries(client, query):
    assert client.get('/?' + query).status_code == 400


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_only_get(client, method):
    response = client.open('/?year=1923', method=method)
    assert response.status_code == 405
    assert response.headers["Allow"] == "GET"


def test_help_no_unfiltered_dump_and_stable_bits(client):
    assert "Browse political quotes" in client.get('/').text
    assert "Quote John Maynard Keynes 0" not in client.get('/').text
    first = client.get('/?year=1923')
    second = client.get('/?year=1923')
    assert len(re.findall(r"ID: [A-Za-z0-9+/]{43}=\n", first.text)) == 5
    assert first.text == second.text
    assert first.headers["Cache-Control"] == "no-store"


def test_author_links_and_year_form(client):
    from bs4 import BeautifulSoup
    from urllib.parse import parse_qsl, urlsplit

    response = client.get('/')
    assert response.content_type == "text/html; charset=utf-8"
    soup = BeautifulSoup(response.text, 'html.parser')
    forms = soup.find_all('form')
    assert len(forms) == 1
    form = forms[0]
    assert form['method'] == 'get' and form['action'] == '/'
    assert [field['name'] for field in form.select('input[name]')] == ['year']
    assert form.select_one('input')['required'] == ''
    assert client.get('/?year=1923').text.count('\nAuthor: ') == 5
    links = [a['href'] for a in soup.select('a[href^="/?"]')]
    assert '/?name=John+Maynard+Keynes' in links
    assert '/?year=1923&name=John+Maynard+Keynes' in links
    for link in links:
        keys = [key for key, _ in parse_qsl(urlsplit(link).query)]
        assert 'name' in keys  # No standalone per-year list.
        assert keys == sorted(keys, reverse=True)
        result = client.get(link)
        assert result.status_code == 200
        assert result.content_type == 'text/plain; charset=utf-8'
        assert '\nAuthor: ' in result.text


def test_bits_stable_across_restarts_and_depend_on_salt(monkeypatch):
    monkeypatch.setenv("QUOTE_BITS_SALT", "test-salt")
    first = create_app().test_client().get('/?name=John%20Keynes').text
    restarted = create_app().test_client().get('/?name=John%20Keynes').text
    assert first == restarted
    monkeypatch.setenv("QUOTE_BITS_SALT", "different-salt")
    changed = create_app().test_client().get('/?name=John%20Keynes').text
    assert re.findall(r"ID: ([A-Za-z0-9+/]{43}=)", first) != re.findall(r"ID: ([A-Za-z0-9+/]{43}=)", changed)
    assert re.sub(r"ID: [A-Za-z0-9+/]{43}=", "", first) == re.sub(r"ID: [A-Za-z0-9+/]{43}=", "", changed)


def test_bits_depend_on_quote_text():
    from quote_server.app import quote_bits
    assert quote_bits("First quote", "salt") != quote_bits("Second quote", "salt")
    import base64
    import hashlib
    encoded = quote_bits("First quote", "salt")
    assert base64.b64decode(encoded, validate=True) == hashlib.sha256(b"salt\0First quote").digest()


def test_missing_dataset(tmp_path):
    client = create_app(tmp_path / "absent.json").test_client()
    assert client.get('/?year=1923').status_code == 503
