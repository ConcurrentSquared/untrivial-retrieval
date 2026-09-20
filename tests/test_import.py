from quote_server.import_wikiquote import extract


def test_sources_dates_and_exclusions():
    page = {"title": "Example", "revid": 42, "text": '''
    <h2>Quotes</h2><h3>Book (1923)</h3>
    <ul><li>First quote<ul><li>Chapter 1</li></ul></li>
    <li>Second quote<ul><li>Speech (1924)<ul><li>A later comment (2001)</li></ul></li></ul></li>
    <li>Ambiguous<ul><li>Speech 1924, reprinted 1950</li></ul></li>
    <li>Without source</li></ul>
    <h2>Quotes</h2><ul><li>Undated<ul><li>Some book</li></ul></li></ul>
    <h2>Quotes about Example</h2><ul><li>Other person's quote<ul><li>Book (1925)</li></ul></li></ul>
    <h2>Misattributed</h2><ul><li>Not theirs<ul><li>Book (1926)</li></ul></li></ul>
    '''}
    rows = extract(page)
    assert [(r["text"], r["year"]) for r in rows] == [("First quote", 1923), ("Second quote", 1924)]
    assert rows[0]["year_evidence"] == "Book (1923)"
    assert "oldid=42" in rows[0]["url"]
    assert "fair use" in rows[0]["license"]


def test_page_numbers_are_not_years():
    from quote_server.import_wikiquote import candidate_years
    assert candidate_years('Volume II, p. 1923') == set()
    assert candidate_years('Hansard, vol. 346, col. 2139.') == set()
    assert candidate_years('Book (1923), pp. 1924–1925') == {1923}
    page = {'title': 'Example', 'revid': 42, 'text': '''
    <h2>Quotes</h2><h3>Speech (1940)</h3>
    <ul><li>Quote<ul><li>Hansard, col. 1941.</li></ul></li></ul>'''}
    assert extract(page)[0]['year'] == 1940
