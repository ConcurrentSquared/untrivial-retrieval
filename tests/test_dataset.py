"""Verify the shipped snapshot, rather than just the synthetic API fixtures."""
import json
from pathlib import Path

from quote_server.app import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_covers_entire_roster():
    snapshot = json.loads((ROOT / 'quote_server/data/quotes.json').read_text())
    authors = (ROOT / 'quote_server/data/authors.txt').read_text().splitlines()
    assert len(authors) == len(set(authors)) == 100
    assert set(snapshot['coverage']) == set(authors)
    assert all(count > 0 for count in snapshot['coverage'].values())
    assert not snapshot['errors']
    assert sum(snapshot['coverage'].values()) == len(snapshot['quotes'])
    for row in snapshot['quotes']:
        for key in ('author', 'text', 'source', 'year_evidence', 'url', 'attribution', 'license'):
            assert row[key]
        assert isinstance(row['year'], int)
        assert 'oldid=' in row['url']
    client = create_app().test_client()
    for author in authors:
        response = client.get('/', query_string={'name': author})
        assert response.status_code == 200
        assert 1 <= response.text.count('\nAuthor: ') <= 5


def test_real_keynes_example():
    client = create_app().test_client()
    full = client.get('/?year=1923&name=John%20Keynes')
    assert full.status_code == 200
    assert full.text.count('\nAuthor: ') > 5
    assert client.get('/?name=John%20Keynes').text.count('\nAuthor: ') == 5
    assert client.get('/?year=1923').text.count('\nAuthor: ') == 5
