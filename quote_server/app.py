"""Plain-text, order-sensitive quotation retrieval. Run: flask --app quote_server.app run."""
import hashlib
from collections import Counter, defaultdict
import base64
import json
import os
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlencode

from flask import Flask, Response, render_template, request

DATA = Path(__file__).parent / "data" / "quotes.json"
DEFAULT_BITS_SALT = "untrivial-retrieval-v1"
HELP = ('Use /?year=1923&name=John%20Maynard%20Keynes\n'
        'Parameters must be in reverse alphabetical order: year, name.\n'
        'Both filters: all matching quotes. One filter: first five by year, author, page order.\n')


def quote_bits(text, salt):
    digest = hashlib.sha256(salt.encode("utf-8") + b"\0" + text.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def create_app(data_path=None):
    app = Flask(__name__, static_folder=None)
    bits_salt = os.environ.get("QUOTE_BITS_SALT", DEFAULT_BITS_SALT)
    path = Path(data_path or os.environ.get("QUOTES_FILE", DATA))
    records = json.loads(path.read_text(encoding="utf-8"))["quotes"] if path.exists() else None
    if records is not None:
        for row in records:
            if not row.get("author") or type(row.get("year")) is not int or not 1 <= row["year"] <= 9999:
                raise ValueError("Every quote must have an author and a valid integer year")
        records.sort(key=lambda q: (q["year"], q["author"].casefold(), q["position"]))

    def plain(text, status=200):
        return Response(text, status=status, content_type="text/plain; charset=utf-8",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.before_request
    def get_only():
        if request.method != "GET":
            response = plain("Only HTTP GET is supported.\n", 405)
            response.headers["Allow"] = "GET"
            return response

    @app.errorhandler(404)
    def missing(_):
        return plain("Not found.\n", 404)

    def home():
        authors = defaultdict(Counter)
        for row in records or []:
            authors[row["author"]][row["year"]] += 1
        years = sorted({row["year"] for row in records or []})

        def search_link(name, year=None):
            pairs = [("year", year)] if year is not None else []
            pairs.append(("name", name))
            return "/?" + urlencode(pairs)

        return Response(render_template(
            "index.html", authors=sorted(authors.items(), key=lambda item: item[0].casefold()),
            years=years, search_link=search_link,
            quote_count=len(records or []), dataset_missing=records is None,
        ), content_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.route("/", methods=["GET"], provide_automatic_options=False)
    def quotes():
        try:
            raw = request.query_string.decode("ascii")
            if re.search(r"%(?![0-9a-fA-F]{2})", raw):
                raise ValueError("Malformed percent encoding")
            pairs = parse_qsl(raw, keep_blank_values=True, strict_parsing=True,
                              encoding="utf-8", errors="strict", max_num_fields=2)
        except (ValueError, UnicodeError):
            return plain("Invalid query string.\n" + HELP, 400)
        keys = [key for key, _ in pairs]
        if any(key not in {"year", "name"} for key in keys) or len(set(keys)) != len(keys):
            return plain("Only unique year and name parameters are allowed.\n" + HELP, 400)
        if keys != sorted(keys, reverse=True):
            return plain("Parameters must appear in reverse alphabetical order: year before name.\n", 400)
        if not pairs:
            return home()
        params = dict(pairs)
        if any(not value.strip() for value in params.values()):
            return plain("Supplied parameters must not be blank; omit an unused filter.\n", 400)
        if "year" in params and (not re.fullmatch(r"[0-9]{1,4}", params["year"]) or int(params["year"]) < 1):
            return plain("year must be an integer from 1 to 9999.\n", 400)
        if records is None:
            return plain("Dataset missing. Run: python -m quote_server.import_wikiquote\n", 503)
        name = params.get("name", "").strip().strip('"').casefold()
        if "name" in params and not name:
            return plain("name must not be blank.\n", 400)
        if name == "john keynes":
            name = "john maynard keynes"
        matches = [q for q in records
                   if ("name" not in params or name in
                       {value.casefold() for value in [q["author"], *q.get("aliases", [])]})
                   and ("year" not in params or q["year"] == int(params["year"]))]
        if len(params) == 1:
            matches = matches[:5]
        if not matches:
            return plain("No matching quotes.\n")
        blocks = []
        for q in matches:
            source_links = "\nSource links: " + " ".join(q["source_urls"]) if q.get("source_urls") else ""
            blocks.append(f'{q["text"]}\n\nAuthor: {q["author"]}\nYear: {q["year"]}'
                          f'\nYear evidence: {q["year_evidence"]}\nSource: {q["source"]}{source_links}'
                          f'\nWikiquote: {q["url"]}\nAttribution: {q["attribution"]}'
                          f'\nLicense: {q["license"]}\nID: {quote_bits(q["text"], bits_salt)}')
        return plain("\n\n---\n\n".join(blocks) + "\n")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
