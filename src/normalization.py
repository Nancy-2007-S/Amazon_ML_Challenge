"""
normalization.py
----------------
Business Entity Resolution — Stage 2: Normalization

Provides three pure functions:
  normalize_name(name_str)    -> str | None
  normalize_address(addr_str) -> str | None
  normalize_country(c_str)    -> str | None

And two helpers for DataFrames:
  normalize_dataframe(df)         -> df with _norm columns added (in-place, no copy)
  compute_norm_metrics_chunked()  -> writes metrics JSON, one source at a time

Design goals:
  - CPU-only, no GPU dependencies
  - Memory-safe: processes one source file at a time; uses chunked I/O for stats
  - No external APIs, no translation models, no internet lookups
  - NFKC Unicode (preserves Indic combining marks)
  - Country treated as open-set (never hard-coded)
  - Raw columns are NEVER modified; only _norm columns are added
"""

import unicodedata
import json
import os
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────

# Literal strings that should be treated as missing values
_NULL_LITERALS = frozenset(['nan', 'null', 'none', ''])

# Legal suffix normalizations (name only)
_NAME_SUFFIX_MAP = {
    'limited': 'ltd',
    'private': 'pvt',
    'corporation': 'corp',
    'company': 'co',
    'incorporated': 'inc',
}

# Street-type abbreviations (address only).
# NOTE: We intentionally exclude state full-name -> abbreviation mappings
# (e.g., 'ohio' -> 'oh') because state names also appear mid-string as
# street names ('GEORGIA AVE', 'OHIO DRIVE'), causing false substitutions.
# State abbreviations already appear natively in the data after lowercasing.
_ADDR_TYPE_MAP = {
    'street':    'st',
    'avenue':    'ave',
    'road':      'rd',
    'lane':      'ln',
    'court':     'ct',
    'drive':     'dr',
    'apartment': 'apt',
    'highway':   'hwy',
    'building':  'bldg',
}

# ── core utility ───────────────────────────────────────────────────────────────

def _remove_punctuation_unicode(s: str) -> str:
    """
    Replace non-letter/number/mark/space Unicode characters with a space.

    Uses Unicode General Category codes so that Indic combining marks
    (vowel signs, matras — category 'M') are preserved intact:
      L = Letters (all scripts)
      N = Numbers
      M = Marks (combining diacritics / vowel signs — critical for Devanagari,
                 Gujarati, Tamil, etc.)
      Z = Separators (whitespace)

    Everything else (punctuation, symbols, control chars) becomes a space.

    Why not re.sub(r'[^a-z0-9\s]', ' ', s)?
      That discards any non-ASCII byte, corrupting all Indic script characters.
    """
    out = []
    for ch in s:
        if unicodedata.category(ch)[0] in ('L', 'N', 'M', 'Z'):
            out.append(ch)
        else:
            out.append(' ')
    return ''.join(out)


# ── public normalization functions ─────────────────────────────────────────────

def normalize_name(name_str) -> str | None:
    """
    Normalize a business name string.

    Steps (in order):
      1. Guard: return None for NaN / null literals / empty strings.
      2. NFKC Unicode normalization — composes decomposed characters safely;
         flattens compatibility variants (e.g., ＬＬＣ -> LLC).
      3. Lowercase.
      4. Strip .com domain suffixes (e.g., 'zvalibaba.com' -> 'zvalibaba').
      5. Collapse dotted acronyms before punctuation removal so they become
         single tokens: l.l.c. -> llc, l.t.d. -> ltd, p.v.t. -> pvt, inc. -> inc.
      6. Remove common prefixes: m/s (Messrs), c/o (care of).
      7. Normalise & -> ' and '.
      8. Strip punctuation while preserving Unicode combining marks (see above).
      9. Token-level suffix normalisation: 'limited' -> 'ltd', etc.
     10. Collapse whitespace; return None if result is empty.

    Does NOT:
      - Transliterate Indic scripts to English.
      - Use external databases or APIs.
      - Modify word order (handled by token-overlap features in Stage 5).
      - Resolve typos (handled by edit-distance features in Stage 5).
    """
    if pd.isna(name_str):
        return None
    s = str(name_str).strip()
    if s.lower() in _NULL_LITERALS:
        return None

    # Steps 2–3
    s = unicodedata.normalize('NFKC', s).lower()

    # Step 4: domain cleanup
    s = s.replace('.com', '')

    # Step 5: collapse dotted acronyms BEFORE punctuation removal
    s = s.replace('l.l.c.', 'llc')
    s = s.replace('l.t.d.', 'ltd')
    s = s.replace('p.v.t.', 'pvt')
    s = s.replace('inc.',   'inc')

    # Step 6: common prefixes
    s = s.replace('m/s', ' ')
    s = s.replace('c/o', ' ')

    # Step 7: ampersand
    s = s.replace('&', ' and ')

    # Step 8: strip punctuation (Unicode-safe)
    s = _remove_punctuation_unicode(s)

    # Step 9: suffix tokens
    tokens = s.split()
    tokens = [_NAME_SUFFIX_MAP.get(t, t) for t in tokens]

    result = ' '.join(tokens)
    return result if result else None


def normalize_address(addr_str) -> str | None:
    """
    Normalize a business address string.

    Steps (in order):
      1. Guard: return None for NaN / null literals / empty strings.
         Literal 'NULL' and 'nan' appearing as address text are treated as missing.
      2. NFKC Unicode normalization + lowercase.
      3. Normalise & -> ' and '.
      4. Strip punctuation (Unicode-safe, preserves Indic combining marks).
      5. Strip leading zeros from purely numeric tokens
         (e.g., '00106' -> '106').
         Safe because: only applied when the entire token is digits.
         NOT applied to: '4th', '02-14', postal codes with letters, etc.
      6. Token-level street-type abbreviation (street->st, avenue->ave, etc.).
      7. Collapse whitespace; return None if result is empty.

    Does NOT:
      - Map state full names to abbreviations (ambiguous: 'GEORGIA AVE').
      - Transliterate Indic scripts.
      - Use external geocoding or address parsers.
    """
    if pd.isna(addr_str):
        return None
    s = str(addr_str).strip()
    if s.lower() in _NULL_LITERALS:
        return None

    # Steps 2: NFKC + lowercase
    s = unicodedata.normalize('NFKC', s).lower()

    # Step 3: ampersand
    s = s.replace('&', ' and ')

    # Step 4: strip punctuation (Unicode-safe)
    s = _remove_punctuation_unicode(s)

    # Steps 5–6: per-token transformations
    tokens = s.split()
    out = []
    for t in tokens:
        # Step 5: leading-zero strip on pure numeric tokens
        if t.isdigit():
            t = t.lstrip('0') or '0'
        # Step 6: street-type abbreviation
        t = _ADDR_TYPE_MAP.get(t, t)
        out.append(t)

    result = ' '.join(out)
    return result if result else None


def normalize_country(c_str) -> str | None:
    """
    Normalize a country string.

    Simply lowercases and strips whitespace.
    Country is treated as an OPEN-SET field — no hard-coded list.
    The test set introduces France (and potentially others), so we must never
    one-hot or filter on known country values.
    """
    if pd.isna(c_str):
        return None
    s = str(c_str).strip().lower()
    return s if s not in _NULL_LITERALS else None


# ── DataFrame helper ───────────────────────────────────────────────────────────

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a DataFrame IN-PLACE (no full copy).

    Adds:
      business_name_norm    — from normalize_name()
      business_address_norm — from normalize_address()

    Raw columns (business_name, business_address, country) are NEVER touched.

    Memory note:
      .apply() processes row-by-row; peak extra RAM ≈ size of two new string
      columns, typically 0.5–0.8× the size of the source columns.
      On an i3 laptop this is safe for chunks up to ~200k rows.
    """
    df['business_name_norm'] = df['business_name'].apply(normalize_name)
    df['business_address_norm'] = df['business_address'].apply(normalize_address)
    return df


# ── chunked metrics computation (memory-safe for i3) ──────────────────────────

def compute_norm_metrics_chunked(
    source_files: list[str],
    output_json: str,
    chunk_size: int = 200_000,
    verbose: bool = True,
) -> dict:
    """
    Compute normalization metrics for each source file USING CHUNKED I/O.

    Loads only `chunk_size` rows into memory at a time per file.
    Peak RAM per chunk (200k rows × 4 string cols) ≈ 150–250 MB — safe on i3.

    Loading the full S2 (5M rows) at once would require ~1.5–2 GB RAM, which
    is risky on an i3 laptop. This chunked approach stays under ~300 MB peak.

    Parameters
    ----------
    source_files : list of absolute file paths to TSV source files
    output_json  : path to write the metrics JSON file
    chunk_size   : rows per chunk (default 200k ≈ safe for 4 GB RAM laptops)
    verbose      : print progress to stdout

    Returns
    -------
    dict : metrics keyed by filename
    """
    metrics = {}

    for fpath in source_files:
        fname = os.path.basename(fpath)
        if verbose:
            print(f"Processing {fname} in chunks of {chunk_size:,}...")

        row_count = 0
        raw_name_missing = 0
        norm_name_missing = 0
        raw_addr_missing = 0
        norm_addr_missing = 0
        raw_name_set = set()
        norm_name_set = set()
        raw_addr_set = set()
        norm_addr_set = set()

        reader = pd.read_csv(fpath, sep='\t', chunksize=chunk_size,
                             dtype=str, keep_default_na=False)

        for chunk in reader:
            row_count += len(chunk)

            # Replace literal null strings with actual NaN for isna() detection
            chunk.replace({'nan': pd.NA, 'null': pd.NA,
                           'NULL': pd.NA, 'None': pd.NA, '': pd.NA},
                          inplace=True)

            # Raw missing counts
            raw_name_missing += chunk['business_name'].isna().sum()
            raw_addr_missing += chunk['business_address'].isna().sum()

            # Unique raw values (accumulate in sets — exact but uses memory)
            raw_name_set.update(chunk['business_name'].dropna().unique())
            raw_addr_set.update(chunk['business_address'].dropna().unique())

            # Apply normalization
            chunk['business_name_norm'] = chunk['business_name'].apply(normalize_name)
            chunk['business_address_norm'] = chunk['business_address'].apply(normalize_address)

            # Norm missing counts
            norm_name_missing += chunk['business_name_norm'].isna().sum()
            norm_addr_missing += chunk['business_address_norm'].isna().sum()

            # Unique norm values
            norm_name_set.update(chunk['business_name_norm'].dropna().unique())
            norm_addr_set.update(chunk['business_address_norm'].dropna().unique())

            if verbose:
                print(f"  ... {row_count:,} rows processed", end='\r')

        if verbose:
            print(f"  Done: {row_count:,} rows            ")

        metrics[fname] = {
            "row_count":         row_count,
            "raw_name_missing":  int(raw_name_missing),
            "norm_name_missing": int(norm_name_missing),
            "raw_addr_missing":  int(raw_addr_missing),
            "norm_addr_missing": int(norm_addr_missing),
            "raw_name_unique":   len(raw_name_set),
            "norm_name_unique":  len(norm_name_set),
            "raw_addr_unique":   len(raw_addr_set),
            "norm_addr_unique":  len(norm_addr_set),
        }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    if verbose:
        print(f"\nMetrics written to {output_json}")

    return metrics


# ── smoke tests (safe on i3 — no full file loaded) ────────────────────────────

def run_smoke_tests(out_path: str):
    """
    Run all required before/after normalization examples.
    Does NOT load any TSV file — safe on any machine.
    """
    results = {}

    # Multilingual integrity
    multilingual = [
        "JAMUI, बिहार",
        "Vesu, ગુજરાત",
        "ગુજરાતી",
        "हिंदी",
        "महाराष्ट्र",
    ]
    for t in multilingual:
        results[f"multilingual_addr|{t}"] = (t, normalize_address(t))

    # Numeric normalization
    numeric = [
        "Apt 04, Unit 007",
        "Zip 02138",
        "1/2",
        "12-14",
        "4th floor",
        "00106 GEORGIA AVE",
    ]
    for n in numeric:
        results[f"numeric_addr|{n}"] = (n, normalize_address(n))

    # Canonical edge cases
    edge = [
        ("LLC/L.L.C. name",   "name", "Crooks National International L.L.C.", normalize_name),
        ("LLC/L.L.C. name2",  "name", "Crooks National International LLC",    normalize_name),
        ("Avenue addr",        "addr", "4849 MANZANITA AVENUE",                normalize_address),
        ("OH/Ohio addr",       "addr", "689 Olde Orchard Court, Columbus, Ohio", normalize_address),
        ("OH/Ohio addr2",      "addr", "689 Olde Orchard Court, Columbus, OH",   normalize_address),
        ("Typo name",          "name", "BJ Quraimty Zhejiang LLC",             normalize_name),
        ("URL/domain name",    "name", "ZVALIBABA.COM",                        normalize_name),
        ("Word-order name1",   "name", "Pvt Rani Land Ltd.",                   normalize_name),
        ("Word-order name2",   "name", "Rani Land Pvt Ltd.",                   normalize_name),
        ("NULL addr literal",  "addr", "NULL",                                 normalize_address),
        ("nan addr literal",   "addr", "nan",                                  normalize_address),
        ("Hindi addr inline",  "addr", "PLOT 506 NEAR SBI BANK MAIN ROAD, JAMUI, बिहार", normalize_address),
        ("Gujarati addr inline","addr","Shop No. M-2693, Vesu, ગુજરાત",       normalize_address),
        ("MS prefix name",     "name", "M/s Alpas Investments",               normalize_name),
        ("Ampersand name",     "name", "A & B Corporation",                   normalize_name),
    ]
    for tag, kind, raw, fn in edge:
        results[f"edge_{kind}|{tag}"] = (raw, fn(raw))

    # DBA case — explicitly documented as unresolvable by normalization
    results["UNRESOLVABLE|DBA trade name"] = (
        "BJ Quality Zhejiang LLC  <->  Nexkor",
        "WILL NOT MATCH — requires address-based features in Stage 5"
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        for key, (raw, norm) in results.items():
            f.write(f"[{key}]\n  RAW : {raw}\n  NORM: {norm}\n\n")

    return results


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_DIR  = "dataset/train/"
    OUT_DIR   = "output/"

    SOURCES = [
        DATA_DIR + "train_source1.tsv",
        DATA_DIR + "train_source2.tsv",
        DATA_DIR + "train_source3.tsv",
    ]

    print("=" * 60)
    print("MEMORY SAFETY NOTICE")
    print("=" * 60)
    print("Full dataset sizes (approx):")
    print("  S1: 2.2M rows x 4 str cols  --> ~600 MB RAM if loaded whole")
    print("  S2: 5.0M rows x 4 str cols  --> ~1.4 GB RAM if loaded whole")
    print("  S3: 5.3M rows x 4 str cols  --> ~1.5 GB RAM if loaded whole")
    print()
    print("This script uses CHUNKED I/O (200k rows/chunk).")
    print("Peak RAM per chunk: ~150-250 MB  --> SAFE on i3 laptop.")
    print("Total comfortable RAM required: ~300 MB.")
    print("=" * 60)
    print()

    # Step 1: smoke tests (no file I/O — safe everywhere)
    print("Running smoke tests (no TSV loaded)...")
    run_smoke_tests(OUT_DIR + "norm_smoke_tests.txt")
    print(f"  Smoke test output: {OUT_DIR}norm_smoke_tests.txt")
    print()

    # Step 2: chunked metrics over all 3 sources
    print("Computing metrics with chunked I/O...")
    metrics = compute_norm_metrics_chunked(
        source_files=SOURCES,
        output_json=OUT_DIR + "norm_metrics_all.json",
        chunk_size=200_000,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for fname, m in metrics.items():
        name_collapse = m['raw_name_unique'] - m['norm_name_unique']
        addr_collapse = m['raw_addr_unique'] - m['norm_addr_unique']
        print(f"\n{fname}")
        print(f"  Rows            : {m['row_count']:>10,}")
        print(f"  Name missing    : raw={m['raw_name_missing']:,}  norm={m['norm_name_missing']:,}")
        print(f"  Addr missing    : raw={m['raw_addr_missing']:,}  norm={m['norm_addr_missing']:,}")
        print(f"  Unique names    : raw={m['raw_name_unique']:,}  norm={m['norm_name_unique']:,}  (collapsed {name_collapse:,})")
        print(f"  Unique addresses: raw={m['raw_addr_unique']:,}  norm={m['norm_addr_unique']:,}  (collapsed {addr_collapse:,})")
