#!/usr/bin/env python3
"""
scrape.py — Download cannabis strain images from the shops in url_list.txt,
convert them to WebP, and save each one named after its strain.

Currently supported source:
  * can-doc.de  — a standard Shopify store. We read its /products.json API,
    which returns clean product titles and image URLs (real strain photos).

Deliberately skipped:
  * shop.dransay.com — a custom JavaScript (Hydrogen/Remix) app. Its static
    HTML only exposes generic placeholder images; the real per-strain photos
    are rendered client-side and would require a headless browser (Playwright)
    to capture. If you want it added later, that's the path.

Filenames use only the STRAIN NAME, with the vendor, THC/CBD ratio and product
codes stripped out. e.g.  "Omg 30/1 IML PLP Platinum Pave" -> "platinum-pave.webp"

Usage:
    python scrape.py                 # scrape every supported URL in url_list.txt
    python scrape.py --limit 20      # only the first 20 products (quick test)
    python scrape.py --keep-original # also keep the source image next to the webp
    python scrape.py --out mydir     # output directory (default: ./in)

By default images that already exist are skipped (re-runs are incremental and
never overwrite). Pass --overwrite to re-download and replace them.

Dependencies:
    pip install Pillow
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

from PIL import Image

USER_AGENT = "Mozilla/5.0 (compatible; strain-image-scraper/1.0)"
REQUEST_TIMEOUT = 30
RETRIES = 3
POLITE_DELAY = 0.3  # seconds between requests, to be a good citizen


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def http_get(url: str) -> bytes:
    """GET a URL with retries and a browser-like User-Agent."""
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_err = exc
            if attempt < RETRIES:
                time.sleep(attempt)  # simple backoff
    raise RuntimeError(f"GET failed after {RETRIES} tries: {url} ({last_err})")


# --------------------------------------------------------------------------- #
# Strain-name extraction
# --------------------------------------------------------------------------- #
RATIO_TOK = re.compile(r"^\d{1,2}[:/]\d{1,2}$")   # 30/1, 26:01
NUM = re.compile(r"^\d{1,3}$")                     # bare 30, 01, 10
CODE_TOK = re.compile(r"^[A-Za-z0-9#]{1,5}$")

# Filler / unit words that surround the ratio and are never part of a strain name.
FILLER = {"thc", "cbd", "ku", "ku.", "no.", "no", "smalls", "regular", "drop", "med"}

# Short, real strain words that must never be discarded as if they were codes.
KEEP_SHORT = {"og", "x", "gg", "z", "ak", "gsc", "gdp", "mac", "pie", "goo",
              "gas", "sky", "sf", "red", "pop", "cake", "kush", "gelato"}


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _acronym_of_prefix(code: str, words: list[str]) -> bool:
    """True if `code`'s letters are the initials of (or a subsequence within)
    the strain words that follow it, e.g. 'Plp' -> 'Platinum Pave'."""
    cl = _letters(code)
    if not cl:
        return False
    initials = "".join(w[0].lower() for w in words if w[:1].isalpha())
    if initials.startswith(cl):
        return True
    joined = _letters("".join(words[: max(len(cl), 3)]))
    it = iter(joined)
    return len(cl) >= 2 and all(ch in it for ch in cl)


def _looks_like_code(tok: str) -> bool:
    """A short token that reads like a vendor/product abbreviation rather than
    a real word (Plp, Csf, TMS, Sf4, E85, Pcb ...)."""
    letters = _letters(tok)
    if not letters or tok.lower() in KEEP_SHORT:
        return False
    if any(c.isdigit() for c in tok):          # Sf4, E85, G13, Ps3, FaZ9
        return len(tok) <= 4
    if tok.isupper() and len(tok) <= 4:        # CSF, CZE, LS, TMS, GJY, ONO
        return True
    vowels = sum(c in "aeiou" for c in letters)
    if len(letters) <= 4 and vowels == 0:      # Plp, Crf, Sblm, Pcb, Kblt
        return True
    return False


def extract_strain(title: str) -> str:
    """Best-effort extraction of the bare strain name from a shop product title.

    Handles two common shapes:
      "<Vendor> <ratio>: <Strain>"      -> take text after the colon
      "<Vendor> <ratio> <CODES> <Strain>" -> strip vendor+ratio+codes prefix
    """
    t = title.strip()

    # Protect numeric ratios (26:01) so their colon does not trigger a split,
    # then split on any *real* separating colon — the strain sits after it.
    protected = re.sub(r"(\d)[:](\d)", r"\1/\2", t)
    if ":" in protected:
        after = protected.split(":")[-1].strip()
        if after and not after[0].isdigit():
            return after

    tokens = protected.split()
    n = len(tokens)
    i = 0

    # Phase 1 — skip the vendor + THC/CBD ratio prefix (everything up to and
    # including the numeric ratio block).
    passed_numbers = False
    while i < n:
        tok = tokens[i]
        low = tok.lower().strip(".,")
        if RATIO_TOK.match(tok) or NUM.match(tok):
            passed_numbers = True
            i += 1
        elif low in FILLER:
            i += 1
        elif not passed_numbers:
            i += 1
        else:
            break

    # Phase 2 — drop leading product codes (acronyms of the strain, digit codes,
    # filler units, and consonant-only abbreviations).
    while i < n - 1:
        cand = tokens[i]
        low = cand.lower().strip(".,")
        rest = tokens[i + 1:]
        if low in FILLER:
            i += 1
        elif CODE_TOK.match(cand) and _acronym_of_prefix(cand, rest):
            i += 1
        elif _looks_like_code(cand):
            i += 1
        else:
            break

    strain = " ".join(tokens[i:]).strip(" .,-")
    if not strain:                              # never return empty
        strain = " ".join(title.split()[-2:])
    return strain


def slugify(name: str) -> str:
    """Filesystem-safe, lowercase, hyphenated slug: 'Platinum Pavé' -> 'platinum-pave'."""
    # Normalise a few common non-ASCII letters, then strip the rest.
    replacements = {
        "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "é": "e", "è": "e", "ê": "e", "á": "a", "à": "a", "í": "i",
        "ó": "o", "ú": "u", "ñ": "n", "ç": "c",
    }
    s = name.lower()
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "strain"


# --------------------------------------------------------------------------- #
# Source: can-doc.de (Shopify products.json)
# --------------------------------------------------------------------------- #
def is_candoc(url: str) -> bool:
    return "can-doc.de" in urllib.parse.urlparse(url).netloc


def candoc_products(base_url: str):
    """Yield (strain_name, image_url) for every product in a can-doc.de store.

    Uses the public Shopify /products.json endpoint, paginated 250 at a time.
    """
    root = f"{urllib.parse.urlparse(base_url).scheme}://{urllib.parse.urlparse(base_url).netloc}"
    page = 1
    while True:
        api = f"{root}/products.json?limit=250&page={page}"
        try:
            data = json.loads(http_get(api))
        except RuntimeError as exc:
            print(f"  ! failed to fetch page {page}: {exc}", file=sys.stderr)
            break
        products = data.get("products", [])
        if not products:
            break
        for p in products:
            images = p.get("images") or []
            if not images:
                continue
            title = (p.get("title") or "").strip()
            if not title:
                continue
            strain = extract_strain(title)
            yield strain, images[0]["src"], title
        page += 1
        time.sleep(POLITE_DELAY)


# --------------------------------------------------------------------------- #
# Image download + WebP conversion
# --------------------------------------------------------------------------- #
def save_as_webp(image_bytes: bytes, dest: Path, quality: int = 90) -> None:
    with Image.open(io.BytesIO(image_bytes)) as im:
        # WebP does not support palette / CMYK cleanly; normalise the mode.
        if im.mode in ("P", "CMYK"):
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        im.save(dest, format="WEBP", quality=quality, method=6)


def unique_path(directory: Path, slug: str, ext: str) -> Path:
    """Return a non-colliding path: slug.ext, slug-2.ext, slug-3.ext, ..."""
    candidate = directory / f"{slug}.{ext}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{slug}-{n}.{ext}"
        n += 1
    return candidate


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def read_url_list(path: Path) -> list[str]:
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.lower() == "end":
            continue
        if line.startswith(("http://", "https://")):
            urls.append(line)
    return urls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", default="url_list.txt", help="file of shop URLs")
    ap.add_argument("--out", default="in", help="output directory")
    ap.add_argument("--limit", type=int, default=0,
                    help="max products to process per site (0 = all)")
    ap.add_argument("--quality", type=int, default=90, help="WebP quality 1-100")
    ap.add_argument("--keep-original", action="store_true",
                    help="also save the original downloaded image")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-download images that already exist (default: skip them)")
    args = ap.parse_args()

    urls = read_url_list(Path(args.urls))
    if not urls:
        print(f"No usable URLs found in {args.urls}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_ok = total_fail = total_skipped = total_exists = 0

    for url in urls:
        print(f"\n=== {url} ===")
        if not is_candoc(url):
            print("  - skipped: no supported extractor for this site "
                  "(needs a headless browser; see module docstring).")
            total_skipped += 1
            continue

        count = 0
        for strain, img_url, title in candoc_products(url):
            if args.limit and count >= args.limit:
                break
            count += 1

            slug = slugify(strain)
            # Guard: don't re-scrape or overwrite an image we already have.
            # Skip when <slug>.webp exists (unless --overwrite was passed), which
            # makes re-runs cheap and incremental — no download, no conversion.
            base = out_dir / f"{slug}.webp"
            if base.exists() and not args.overwrite:
                print(f"  skip {title!r}  (already have {base.name})")
                total_exists += 1
                continue

            # --overwrite: reuse the exact existing filename; otherwise pick a
            # non-colliding one so genuinely distinct products don't clobber.
            dest = base if args.overwrite else unique_path(out_dir, slug, "webp")
            try:
                raw = http_get(img_url)
                save_as_webp(raw, dest, quality=args.quality)
                if args.keep_original:
                    ext = Path(urllib.parse.urlparse(img_url).path).suffix or ".img"
                    (dest.with_suffix(ext)).write_bytes(raw)
                print(f"  ok  {title!r}")
                print(f"      -> {dest.name}   (strain: {strain!r})")
                total_ok += 1
            except Exception as exc:  # noqa: BLE001 - keep going on any single failure
                print(f"  FAIL {title!r}: {exc}", file=sys.stderr)
                total_fail += 1
            time.sleep(POLITE_DELAY)

    print(f"\nDone. {total_ok} saved, {total_exists} already existed, "
          f"{total_fail} failed, {total_skipped} site(s) skipped.  "
          f"Output: {out_dir.resolve()}")
    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
