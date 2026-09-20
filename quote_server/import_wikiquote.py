"""Import sourced, dated top-level Wikiquote bullets into a local JSON snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
LICENSE = ("Wikiquote contributor text: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/; "
           "underlying quotations may have separate copyright and be included under fair use. "
           "https://en.wikiquote.org/wiki/Wikiquote:Copyrights")
YEAR = re.compile(r"(?<![\w])([12][0-9]{3})(?![\w])")
LOCATOR = re.compile(r"\b(?:pp?\.|pages?|cols?\.|columns?|vols?\.|volumes?)\s*\d+(?:\s*[-–]\s*\d+)?", re.I)
EXCLUDED = re.compile(r"about|disputed|misattributed|unsourced|attributed|external links|see also|references|bibliography", re.I)


def candidate_years(text):
    # Four-digit page/column numbers are not dates. All roster figures are modern.
    text = LOCATOR.sub("", text)
    return {int(y) for y in YEAR.findall(text) if 1600 <= int(y) <= datetime.now(timezone.utc).year}


def unique_year(text):
    years = candidate_years(text)
    return int(years.pop()) if len(years) == 1 else None


def extract(page):
    soup = BeautifulSoup(page["text"], "html.parser")
    headings = {}
    results = []
    seen = set()
    for node in soup.find_all(["h2", "h3", "h4", "h5", "h6", "li"]):
        if node.name.startswith("h"):
            level = int(node.name[1])
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = node.get_text(" ", strip=True).replace("[edit]", "").strip()
            continue
        if node.find_parent("li") or node.find_parent(["table", "figure", "nav"]):
            continue
        if not headings or any(EXCLUDED.search(h) for h in headings.values()):
            continue
        source_list = node.find(["ul", "ol"], recursive=False)
        if source_list is None:
            continue
        # Only the first source bullet: later bullets often contain commentary/variants.
        source_node = source_list.find("li", recursive=False)
        if source_node is None:
            continue
        source_copy = BeautifulSoup(str(source_node), "html.parser")
        for nested in source_copy.select("ul,ol"):
            nested.decompose()
        source = source_copy.get_text(" ", strip=True)
        if not source or re.search(r"misattributed|unsourced|disputed|misquoted", source, re.I):
            continue
        local = BeautifulSoup(str(node), "html.parser")
        for nested in local.select("ul,ol"):
            nested.decompose()
        text = local.get_text(" ", strip=True)
        if not text or text in seen:
            continue
        # Ambiguous multi-year sources are omitted, never assigned an arbitrary year.
        source_years = candidate_years(source)
        year = unique_year(source)
        evidence = source
        if not source_years:
            for heading in reversed(list(headings.values())):
                if candidate_years(heading):
                    year = unique_year(heading)
                    evidence = heading
                    break
        if year is None:
            continue
        seen.add(text)
        title = page["title"]
        url = "https://en.wikiquote.org/w/index.php?" + urlencode({"title": title, "oldid": page["revid"]})
        results.append({"id": hashlib.sha256((title + text).encode()).hexdigest()[:20],
                        "author": title, "year": year, "text": text,
                        "source": " / ".join(headings.values()) + " — " + source,
                        "source_urls": [a["href"] for a in source_node.select('a[href^="http"]')],
                        "year_evidence": evidence, "position": len(results), "url": url,
                        "attribution": f"Wikiquote contributors, {title}; history: https://en.wikiquote.org/w/index.php?title={quote(title)}&action=history",
                        "license": LICENSE})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-download cached API responses")
    parser.add_argument("--limit", type=int, help="Import only the first N authors for a smoke test")
    args = parser.parse_args()
    authors = (ROOT / "data/authors.txt").read_text().splitlines()
    assert len(authors) == len(set(authors)) == 100
    if args.limit:
        authors = authors[:args.limit]
    cache = ROOT / "data/cache"
    cache.mkdir(exist_ok=True)
    quotes, coverage, errors = [], {}, {}
    for author in authors:
        cached = cache / (hashlib.sha256(author.encode()).hexdigest() + ".json")
        try:
            if args.refresh or not cached.exists():
                url = "https://en.wikiquote.org/w/api.php?" + urlencode({"action": "parse", "page": author,
                       "prop": "text|revid", "format": "json", "formatversion": 2, "redirects": 1})
                for attempt in range(5):
                    try:
                        req = Request(url, headers={"User-Agent": "untrivial-retrieval/0.1 (Wikiquote snapshot importer)"})
                        with urlopen(req, timeout=45) as response:
                            payload = json.load(response)
                        if "error" in payload:
                            raise ValueError(payload["error"])
                        cached.write_text(json.dumps(payload), encoding="utf-8")
                        break
                    except HTTPError as exc:
                        if attempt == 4 or exc.code not in {429, 500, 502, 503, 504}:
                            raise
                        retry = exc.headers.get("Retry-After", "")
                        delay = max(30 * (attempt + 1), int(retry) if retry.isdigit() else 0)
                        print(f"{author}: HTTP {exc.code}; retrying in {delay}s", flush=True)
                        time.sleep(delay)
                    except OSError:
                        if attempt == 4:
                            raise
                        time.sleep(2 ** (attempt + 1))
                time.sleep(3)
            page = json.loads(cached.read_text())["parse"]
            rows = extract(page)
            for row in rows:
                row["aliases"] = [author] if author != row["author"] else []
            quotes.extend(rows)
            coverage[author] = len(rows)
            print(f"{author}: {len(rows)}", flush=True)
        except Exception as exc:
            errors[author] = str(exc)
            print(f"{author}: ERROR {exc}", flush=True)
    payload = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "coverage": coverage,
               "errors": errors, "year_policy": "Unique year in first source bullet, else nearest dated heading; may be publication/reporting year, not utterance year.",
               "quotes": quotes}
    target = ROOT / "data" / ("quotes.sample.json" if args.limit else "quotes.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"Saved {len(quotes)} quotes; {sum(n > 0 for n in coverage.values())}/{len(authors)} authors covered; {len(errors)} errors.")
    if errors or any(n == 0 for n in coverage.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
