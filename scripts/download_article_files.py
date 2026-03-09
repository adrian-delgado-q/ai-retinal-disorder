import argparse
import io
import json
import re
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


DEFAULT_INPUT = Path("data/articles/eye_conditions_open_access_database.json")
DEFAULT_OUTPUT = Path("data/articles/downloads")
DEFAULT_MANIFEST_NAME = "manifest.json"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
COOKIE_JAR = CookieJar()
OPENER = build_opener(HTTPCookieProcessor(COOKIE_JAR))


@dataclass(frozen=True)
class ArticleRecord:
    index: int
    title: str
    url: str


class PdfLinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.download_pdf_hrefs: list[str] = []
        self.pdf_like_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return

        attr_map = {key.lower(): value for key, value in attrs}
        href = attr_map.get("href")
        if not href:
            return

        aria_label = (attr_map.get("aria-label") or "").strip().lower()
        if aria_label == "download pdf":
            self.download_pdf_hrefs.append(href)

        href_lower = href.lower()
        if href_lower.endswith(".pdf") or "/pdf/" in href_lower:
            self.pdf_like_hrefs.append(href)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download article PDFs from the URLs listed in a JSON file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"JSON file containing article objects with a 'url' field. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Single flat folder where PDFs will be written. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Network timeout in seconds for page fetches and downloads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace local PDFs that already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N article URLs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve PDF URLs and print them without downloading.",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help=f"Manifest filename written into the output directory. Default: {DEFAULT_MANIFEST_NAME}",
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("data/articles/eye_conditions_open_access_database_download_status.json"),
        help=(
            "Write per-article download status for the input list to this JSON file. "
            "Default: data/articles/eye_conditions_open_access_database_download_status.json"
        ),
    )
    return parser.parse_args()


def load_articles(path: Path, limit: int | None) -> list[ArticleRecord]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")

    articles: list[ArticleRecord] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        title = str(item.get("title", "")).strip() or f"article_{index:04d}"
        articles.append(ArticleRecord(index=index, title=title, url=url))
        if limit is not None and len(articles) >= limit:
            break
    return articles


def load_input_payload(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return payload


def fetch_response(
    url: str,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> tuple[bytes, object, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    if extra_headers:
        headers.update(extra_headers)
    request = Request(url, headers=headers)
    with OPENER.open(request, timeout=timeout) as response:
        return response.read(), response.headers, response.geturl()


def fetch_text(url: str, timeout: float) -> str:
    payload, headers, _ = fetch_response(url, timeout)
    charset = headers.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def normalize_stem(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "article"


def article_identifier(article: ArticleRecord) -> str:
    parsed = urlparse(article.url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] == "articles":
        return path_parts[1]
    return f"article_{article.index:04d}"


def extract_pmcid(url: str) -> str | None:
    match = re.search(r"/articles/(PMC\d+)", url)
    if match:
        return match.group(1)
    return None


def normalize_oa_href(href: str) -> str:
    # oa.fcgi often returns ftp:// links; HTTPS works for the same host/path.
    if href.startswith("ftp://"):
        return "https://" + href[len("ftp://") :]
    return href


def lookup_pdf_via_oa_api(pmcid: str, timeout: float) -> tuple[str | None, str]:
    oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    payload, _, _ = fetch_response(
        oa_url,
        timeout=timeout,
        extra_headers={"Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8"},
    )
    text = payload.decode("utf-8", errors="replace")
    root = ET.fromstring(text)

    error_node = root.find(".//error")
    if error_node is not None:
        code = error_node.attrib.get("code", "unknown")
        message = (error_node.text or "").strip()
        if code == "idIsNotOpenAccess":
            return None, "not_open_access"
        return None, f"oa_error:{code}:{message}"

    package_href: str | None = None
    # Expected structure: <OA><records><record><link format="pdf" href="..."/></record></records></OA>
    for link in root.findall(".//link"):
        fmt = (link.attrib.get("format") or "").lower()
        href = link.attrib.get("href")
        if not href:
            continue
        normalized = normalize_oa_href(href)
        if fmt == "pdf" or normalized.lower().endswith(".pdf"):
            return normalized, "ok"
        if fmt in {"tgz", "tar.gz"} or normalized.lower().endswith((".tgz", ".tar.gz")):
            package_href = normalized
    if package_href:
        return package_href, "oa_package_tgz"
    return None, "no_pdf_link_in_oa_record"


def derive_pdf_url(article: ArticleRecord, timeout: float) -> str:
    pmcid = extract_pmcid(article.url)
    if pmcid:
        try:
            oa_pdf, oa_status = lookup_pdf_via_oa_api(pmcid, timeout=timeout)
        except (ET.ParseError, HTTPError, URLError, ValueError):
            oa_pdf = None
            oa_status = "oa_lookup_failed"
        if oa_pdf:
            return oa_pdf
        # For PMCID-backed records, prefer explicit OA diagnostics over
        # scraping the site-level PDF endpoint, which is commonly bot-blocked.
        if oa_status == "not_open_access":
            raise ValueError(
                f"PMCID {pmcid} is not in the PMC Open Access file subset "
                "(direct automated PDF download blocked)"
            )
        if oa_status in {"no_pdf_link_in_oa_record", "oa_lookup_failed"} or oa_status.startswith("oa_error:"):
            raise ValueError(
                f"Could not resolve OA PDF for {pmcid} (status={oa_status}); "
                "site-level PDF endpoint is blocked for scripted access in this environment"
            )

    html = fetch_text(article.url, timeout=timeout)
    parser = PdfLinkExtractor()
    parser.feed(html)

    for href in parser.download_pdf_hrefs:
        absolute = urljoin(article.url, href)
        if absolute.lower().endswith(".pdf"):
            return absolute

    for href in parser.download_pdf_hrefs:
        return urljoin(article.url, href)

    for href in parser.pdf_like_hrefs:
        absolute = urljoin(article.url, href)
        if absolute.lower().endswith(".pdf"):
            return absolute

    for href in parser.pdf_like_hrefs:
        return urljoin(article.url, href)

    raise ValueError("No PDF link found on article page")


def filename_from_pdf_url(article: ArticleRecord, pdf_url: str) -> str:
    article_id = article_identifier(article)
    leaf = Path(urlparse(pdf_url).path).name or "article.pdf"
    if not leaf.lower().endswith(".pdf"):
        leaf = f"{leaf}.pdf"
    safe_leaf = normalize_stem(Path(leaf).stem)
    return f"{article.index:04d}_{article_id}_{safe_leaf}.pdf"


def write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_status_report(
    input_payload: list[dict],
    processed_articles: list[ArticleRecord],
    manifest_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    by_index = {int(row["index"]): row for row in manifest_rows if "index" in row}
    processed_indexes = {article.index for article in processed_articles}
    status_rows: list[dict[str, object]] = []

    for index, item in enumerate(input_payload, start=1):
        if not isinstance(item, dict):
            continue
        if index not in processed_indexes:
            continue

        status_row: dict[str, object] = {
            "index": index,
            "title": item.get("title"),
            "url": item.get("url"),
        }
        result = by_index.get(index)
        if result is None:
            status_row["download_status"] = "failed"
            status_row["error"] = "No result recorded"
        else:
            status_row["download_status"] = result.get("status", "unknown")
            status_row["pdf_url"] = result.get("pdf_url")
            status_row["path"] = result.get("path")
            if "error" in result:
                status_row["error"] = result.get("error")
        status_rows.append(status_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(status_rows, indent=2), encoding="utf-8")


def resolve_pdf_from_html(base_url: str, html_payload: bytes) -> str | None:
    charset = "utf-8"
    try:
        html = html_payload.decode(charset, errors="replace")
    except UnicodeDecodeError:
        return None

    parser = PdfLinkExtractor()
    parser.feed(html)

    for href in parser.download_pdf_hrefs:
        absolute = urljoin(base_url, href)
        if absolute.lower().endswith(".pdf"):
            return absolute

    for href in parser.pdf_like_hrefs:
        absolute = urljoin(base_url, href)
        if absolute.lower().endswith(".pdf"):
            return absolute

    return None


def extract_redirect_urls_from_html(base_url: str, html_payload: bytes) -> list[str]:
    html = html_payload.decode("utf-8", errors="replace")
    html_lower = html.lower()
    candidates: list[str] = []

    # Meta refresh: <meta http-equiv="refresh" content="0;url=...">
    meta_match = re.search(
        r'http-equiv=["\']refresh["\'][^>]*content=["\'][^"\']*url=([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if meta_match:
        candidates.append(urljoin(base_url, meta_match.group(1).strip()))

    # Common JS redirects.
    js_patterns = [
        r"""location\.href\s*=\s*["']([^"']+)["']""",
        r"""location\.assign\(\s*["']([^"']+)["']\s*\)""",
        r"""location\.replace\(\s*["']([^"']+)["']\s*\)""",
        r"""window\.open\(\s*["']([^"']+)["']""",
    ]
    for pattern in js_patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            candidates.append(urljoin(base_url, match.strip()))

    # Interstitial pages often require one more request to the same URL
    # after a cookie is set.
    if "preparing to download" in html_lower:
        candidates.append(base_url)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


class GenericLinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value for key, value in attrs}
        for key in ("href", "src", "action", "data-href"):
            value = attr_map.get(key)
            if value:
                self.urls.append(value)


def payload_is_pdf(headers: object, payload: bytes) -> bool:
    content_type = headers.get_content_type().lower()
    if content_type == "application/pdf":
        return True
    return payload.startswith(b"%PDF")


def extract_pdf_from_tar_payload(payload: bytes) -> tuple[bytes, str] | None:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            candidates = [
                member for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(".pdf")
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda member: (len(member.name), member.name))
            chosen = candidates[0]
            handle = archive.extractfile(chosen)
            if handle is None:
                return None
            return handle.read(), Path(chosen.name).name
    except (tarfile.ReadError, OSError):
        return None


def html_snippet(payload: bytes, max_chars: int = 180) -> str:
    text = payload.decode("utf-8", errors="replace")
    compact = " ".join(text.split())
    return compact[:max_chars]


def download_pdf_bytes(
    article_url: str,
    pdf_url: str,
    timeout: float,
) -> tuple[bytes, object, str, str | None]:
    def candidate_score(value: str) -> tuple[int, int]:
        lowered = value.lower()
        score = 0
        if lowered.endswith(".pdf"):
            score += 50
        if "/pdf/" in lowered:
            score += 30
        if "download" in lowered:
            score += 20
        if "prepare" in lowered:
            score -= 10
        return (score, -len(value))

    pending: list[str] = [pdf_url]
    visited: set[str] = set()
    last_headers = None
    last_payload = b""
    max_hops = 8

    while pending and len(visited) < max_hops:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current in {pdf_url, article_url}:
            time.sleep(0.35)

        payload, headers, final_url = fetch_response(
            current,
            timeout=timeout,
            extra_headers={
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
                "Referer": article_url,
            },
        )
        last_headers = headers
        last_payload = payload
        if payload_is_pdf(headers, payload):
            return payload, headers, final_url, None

        extracted = extract_pdf_from_tar_payload(payload)
        if extracted is not None:
            extracted_payload, extracted_name = extracted
            return extracted_payload, headers, final_url, extracted_name

        content_type = headers.get_content_type().lower()
        if not content_type.startswith("text/html"):
            continue

        html_text = payload.decode("utf-8", errors="replace")
        candidates: list[str] = []

        nested_url = resolve_pdf_from_html(final_url, payload)
        if nested_url:
            candidates.append(nested_url)
        candidates.extend(extract_redirect_urls_from_html(final_url, payload))

        generic = GenericLinkExtractor()
        generic.feed(html_text)
        for candidate in generic.urls:
            absolute = urljoin(final_url, candidate)
            if absolute.startswith("http://") or absolute.startswith("https://"):
                candidates.append(absolute)

        # Fallback: pull any quoted http(s) URL from inline JS.
        for match in re.findall(r"""https?://[^"'\s<>]+""", html_text):
            candidates.append(match)

        deduped = []
        seen: set[str] = set()
        for candidate in sorted(candidates, key=candidate_score, reverse=True):
            if candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)

        for candidate in deduped:
            if candidate not in visited and candidate not in pending:
                pending.append(candidate)

    if last_headers is not None:
        raise ValueError(
            f"Expected PDF content, got {last_headers.get_content_type()!r}; "
            f"html_snippet={html_snippet(last_payload)!r}"
        )
    raise ValueError("Expected PDF content but could not fetch response")


def process_article(
    article: ArticleRecord,
    output_dir: Path,
    timeout: float,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, object]:
    pdf_url = derive_pdf_url(article, timeout=timeout)
    filename = filename_from_pdf_url(article, pdf_url)
    destination = output_dir / filename

    record: dict[str, object] = {
        "index": article.index,
        "title": article.title,
        "source_url": article.url,
        "pdf_url": pdf_url,
        "path": str(destination),
    }

    if dry_run:
        record["status"] = "dry_run"
        return record

    if destination.exists() and not overwrite:
        record["status"] = "skipped_existing"
        return record

    try:
        payload, headers, final_url, extracted_name = download_pdf_bytes(
            article_url=article.url,
            pdf_url=pdf_url,
            timeout=timeout,
        )
    except ValueError as error:
        raise ValueError(f"{error} [resolved_url={pdf_url}]") from error

    if extracted_name:
        extracted_safe = normalize_stem(Path(extracted_name).stem)
        filename = f"{article.index:04d}_{article_identifier(article)}_{extracted_safe}.pdf"
        destination = output_dir / filename
        record["path"] = str(destination)
        record["extracted_from_oa_package"] = True
        record["extracted_pdf_name"] = extracted_name

    destination.write_bytes(payload)
    record["status"] = "downloaded"
    record["bytes"] = len(payload)
    record["final_url"] = final_url
    record["content_type"] = headers.get_content_type()
    return record


def main() -> int:
    args = parse_args()
    input_payload = load_input_payload(args.input)
    articles = load_articles(args.input, limit=args.limit)
    if not articles:
        print(f"No article URLs found in {args.input}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    failures = 0

    for article in articles:
        print(f"Processing {article.index}: {article.title}")
        try:
            record = process_article(
                article=article,
                output_dir=args.output_dir,
                timeout=args.timeout,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            manifest_rows.append(record)
            print(f"  {record['status']}: {record['pdf_url']}")
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            failures += 1
            failure_record = {
                "index": article.index,
                "title": article.title,
                "source_url": article.url,
                "status": "failed",
                "error": str(error),
            }
            manifest_rows.append(failure_record)
            print(f"  failed: {error}", file=sys.stderr)

    manifest_path = args.output_dir / args.manifest_name
    write_manifest(manifest_path, manifest_rows)
    write_status_report(
        input_payload=input_payload,
        processed_articles=articles,
        manifest_rows=manifest_rows,
        output_path=args.status_output,
    )

    downloaded = sum(1 for row in manifest_rows if row.get("status") == "downloaded")
    skipped = sum(1 for row in manifest_rows if row.get("status") == "skipped_existing")
    dry_runs = sum(1 for row in manifest_rows if row.get("status") == "dry_run")
    print(
        f"Finished articles={len(articles)} downloaded={downloaded} "
        f"skipped={skipped} dry_run={dry_runs} failures={failures} "
        f"manifest={manifest_path} status_report={args.status_output}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
