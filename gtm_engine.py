#!/usr/bin/env python3
"""
gtm_engine.py — Free LinkedIn-first job-search engine (multi-profile).

Turns the UK sponsor register into a ranked worklist of visa-sponsoring employers
that fit a candidate, with a LinkedIn company link, best-guess profile URLs, and an
AI-drafted connection note + follow-up DM per row. No paid APIs.

Two built-in profiles (select with --profile):
  nandu : B2B SaaS Account Executive -> filters to TECH/SaaS companies, targets
          sales leaders (VP Sales / CRO / Sales Director).
  sister: Project / client management + customer experience -> filters to FASHION,
          RETAIL, MEDIA and AIRLINES/TRAVEL employers, targets CX / client-services /
          PM hiring managers and recruiters.

DATA SOURCES (both free)
  Companies House Public Data API  -> company status, SIC codes, size proxy.
  Anthropic API                    -> the two drafted messages.
  DuckDuckGo (via ddgs, --find-profiles) -> best-guess LinkedIn profile URLs.

USAGE
  python gtm_engine.py --csv register.csv --limit 100 --profile sister --find-profiles
  python gtm_engine.py --csv register.csv --limit 100 --profile nandu
  python gtm_engine.py --csv register.csv --limit 300 --profile sister --no-draft   # list only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    import anthropic
except ImportError:
    anthropic = None


# ---------------------------------------------------------------------------
# COMPANIES HOUSE / GENERAL CONFIG
# ---------------------------------------------------------------------------

CH_BASE = "https://api.company-information.service.gov.uk"
CH_SEARCH_URL = f"{CH_BASE}/search/companies"
CH_PROFILE_URL = f"{CH_BASE}/company"

NAME_MATCH_THRESHOLD = 0.80
LINKEDIN_NOTE_MAX = 300
LINKEDIN_NOTE_TARGET = 180

ANTHROPIC_MODEL = "claude-haiku-4-5"      # cheap + fine for short copy; bump to claude-sonnet-5 for higher quality
ANTHROPIC_MAX_TOKENS = 600

PER_REQUEST_PAUSE = 0.5                    # CH allows 600 req / 5 min
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0

# Size proxy: exclude the tiniest / dormant shells in BROAD mode (no headcount at CH).
MICRO_ACCOUNTS = {"micro-entity", "dormant", "no-accounts"}

LEGAL_SUFFIX_RE = re.compile(r"\b(LIMITED|LTD|LLP|PLC|CIC|C\.I\.C|L\.T\.D)\b\.?", re.I)


# ---------------------------------------------------------------------------
# SIC MAPS  (which companies to KEEP, + friendly sector label + sort priority)
# ---------------------------------------------------------------------------

# --- Profile "nandu": tech / SaaS only ---
TECH_SIC_SECTORS = {
    "62012": "business software development", "62011": "software development",
    "58210": "software publishing (games)", "58290": "software publishing",
    "5821": "software publishing", "5829": "software publishing",
    "62020": "IT consultancy", "62030": "computer facilities management",
    "62090": "IT services", "62": "software / IT services",
    "63110": "data processing / hosting", "63120": "web portals",
    "631": "data / hosting / web", "639": "information services",
    "641": "fintech / banking", "649": "fintech / financial services",
    "661": "fintech / financial services", "662": "fintech / financial services",
    "663": "fintech / fund management", "61": "telecoms",
    "4791": "e-commerce / online retail",
}
TECH_SECTOR_PRIORITY = {
    "business software development": 1, "software development": 1,
    "software publishing": 1, "software publishing (games)": 1,
    "IT services": 2, "data processing / hosting": 2, "web portals": 2,
    "data / hosting / web": 2, "information services": 2, "software / IT services": 2,
    "fintech / banking": 2, "fintech / financial services": 2, "fintech / fund management": 2,
    "computer facilities management": 3, "telecoms": 3,
    "IT consultancy": 4, "e-commerce / online retail": 5,
}

# --- Profile "sister": fashion / retail / media / airlines & travel ---
SISTER_SIC_SECTORS = {
    # fashion & apparel
    "14": "fashion / apparel", "15": "footwear / leather goods", "13": "textiles",
    "7410": "design", "74100": "design",
    # fashion & general retail + e-commerce + wholesale
    "4771": "clothing retail", "4772": "footwear retail", "4751": "textiles retail",
    "4642": "fashion wholesale", "4641": "textiles wholesale",
    "4791": "e-commerce", "47": "retail", "46": "wholesale",
    # media / creative / advertising / PR
    "58": "publishing", "59": "film / TV / music", "60": "broadcasting",
    "6391": "news / media", "73": "advertising / media", "7021": "PR / communications",
    "90": "creative / arts",
    # airlines / aviation / travel / hospitality
    "5110": "airlines", "51": "aviation", "5223": "airport services",
    "79": "travel / tourism", "55": "hospitality",
}
SISTER_SECTOR_PRIORITY = {
    # tier 1 — core target sectors
    "fashion / apparel": 1, "clothing retail": 1, "footwear retail": 1,
    "e-commerce": 1, "airlines": 1, "aviation": 1, "film / TV / music": 1,
    "broadcasting": 1, "advertising / media": 1, "PR / communications": 1, "design": 1,
    # tier 2
    "retail": 2, "publishing": 2, "news / media": 2, "travel / tourism": 2,
    "textiles": 2, "fashion wholesale": 2, "airport services": 2, "creative / arts": 2,
    # tier 3 — adjacent
    "wholesale": 3, "footwear / leather goods": 3, "textiles retail": 3,
    "textiles wholesale": 3, "hospitality": 3,
}


# ---------------------------------------------------------------------------
# CANDIDATES + PROMPTS
# ---------------------------------------------------------------------------

NANDU = {
    "name": "Nandu Padmakumar",
    "line1": "360° Account Executive (B2B SaaS), 5+ years closing complex mid-market and enterprise deals",
    "proof": "112% of annual target ($672k ARR closed against a $600k quota)",
    "strengths": "heavy outbound pipeline generation, multi-threading enterprise deals, and running 30-60 day cycles to close",
}
NANDU_PROMPT = f"""You write LinkedIn outreach for a job candidate reaching out to a \
sales leader who could hire them. You get only the company name, sector, and town — \
NOT the contact's name — so use the literal placeholder {{{{first_name}}}}.

The candidate:
- Name: {NANDU['name']}
- Role: {NANDU['line1']}
- Proof: {NANDU['proof']}
- Strengths: {NANDU['strengths']}

Produce TWO messages.
1) connection_note — HARD LIMIT 300 chars incl. spaces, aim ~{LINKEDIN_NOTE_TARGET}. \
Start "Hi {{{{first_name}}}}," then a hook tied to the company/sector, then ONE light \
credibility signal (a quota/revenue number) as a peer. Its only job is to earn the \
accept: do NOT pitch, ask for a job/call, or mention visa; no emojis; no \
"I hope this finds you well".
2) followup_dm — sent AFTER they accept. Under 90 words. Lead with revenue and the \
numbers (112% attainment, $672k closed). Mention the UK visa need in ONE confident \
line as a minor admin step to land a top-10% revenue producer. End with a soft CTA \
for a brief chat. No fluff.

Only the company name, sector and town are real facts — do NOT invent specifics. \
Return ONLY valid JSON: {{"connection_note": "...", "followup_dm": "..."}}"""

SISTER = {
    "name": "[SISTER_NAME]",          # <-- EDIT THIS to your sister's name
    "quals": "Master's in Project Management and a Bachelor's in Sociology",
    "targets": ("project management, client management, and customer experience / "
                "support — especially in fashion, retail, media, and airlines"),
    "strengths": ("cross-functional delivery, stakeholder coordination, and a "
                  "people-first, customer-empathetic approach"),
    "experience": "",                 # optional: e.g. "with 3 years coordinating delivery at ..."
}
SISTER_PROMPT = f"""You write LinkedIn outreach for a job candidate reaching out to a \
hiring manager or recruiter who could hire them. You get only the company name, \
sector, and town — NOT the contact's name — so use the literal placeholder \
{{{{first_name}}}}.

The candidate:
- Name: {SISTER['name']}
- Qualifications: {SISTER['quals']}
- Target roles: {SISTER['targets']}
- Strengths: {SISTER['strengths']}
- Experience: {SISTER['experience'] or 'early-career; lead with the qualification and strengths, not invented numbers'}

Produce TWO messages.
1) connection_note — HARD LIMIT 300 chars incl. spaces, aim ~{LINKEDIN_NOTE_TARGET}. \
Start "Hi {{{{first_name}}}}," then a hook tied to the company/sector, then ONE \
credibility signal (her Project Management Master's or a relevant strength). Its only \
job is to earn the accept: do NOT pitch, ask for a job/call, or mention visa; no \
emojis; no "I hope this finds you well".
2) followup_dm — sent AFTER they accept. Under 90 words. Lead with her fit for \
project-management, client-management, or customer-experience roles (Master's in \
Project Management + strengths), tied to their sector where natural. Mention the UK \
visa need in ONE confident line as a minor admin step to add a delivery-focused, \
customer-minded team member. End with a soft CTA for a brief chat.

Use ONLY the company name, sector, town, and the candidate facts above — do NOT \
invent metrics, employers, or specifics. If no hard numbers are given, lead with the \
qualification and strengths. Return ONLY valid JSON: \
{{"connection_note": "...", "followup_dm": "..."}}"""

PROFILES = {
    "nandu": {
        "filter_mode": "tech",
        "sic_sectors": TECH_SIC_SECTORS,
        "sector_priority": TECH_SECTOR_PRIORITY,
        "personas": "CRO, VP Sales, Sales Director, Head of Sales, VP Revenue",
        "system_prompt": NANDU_PROMPT,
    },
    "sister": {
        "filter_mode": "broad",
        "sic_sectors": SISTER_SIC_SECTORS,
        "sector_priority": SISTER_SECTOR_PRIORITY,
        "personas": ("Head of Customer Experience, CX Manager, Head of Client Services, "
                     "Client Services Director, Account Director, Project Manager, "
                     "Head of Operations, Talent Acquisition / Recruiter"),
        "system_prompt": SISTER_PROMPT,
    },
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gtm")


# ---------------------------------------------------------------------------
# DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class Lead:
    company: str
    ch_company_name: str = ""
    linkedin_name: str = ""
    website: str = ""
    target_personas: str = ""
    sector: str = ""
    town: str = ""
    accounts_type: str = ""
    google_search: str = ""
    linkedin_company_search: str = ""
    ch_url: str = ""
    company_number: str = ""
    sic_codes: str = ""
    incorporated: str = ""
    status: str = ""
    connection_note: str = ""
    note_chars: Optional[int] = None
    followup_dm: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# STEP 1 — REGISTER FILTERING
# ---------------------------------------------------------------------------

def load_sponsors(csv_path: str, require_a_rating: bool) -> list[str]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if "Route" not in df.columns or "Organisation Name" not in df.columns:
        raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")
    mask = df["Route"].str.strip().str.casefold() == "skilled worker"
    if require_a_rating and "Type & Rating" in df.columns:
        mask &= df["Type & Rating"].str.contains("A rating", case=False, na=False)
    names = (df.loc[mask, "Organisation Name"].str.strip()
             .replace("", pd.NA).dropna().drop_duplicates().tolist())
    log.info("Register: %d rows -> %d unique Skilled Worker sponsors%s",
             len(df), len(names), " (A-rated only)" if require_a_rating else "")
    return names


# ---------------------------------------------------------------------------
# STEP 2 — COMPANIES HOUSE FILTERING (free)
# ---------------------------------------------------------------------------

class CompaniesHouse:
    def __init__(self, api_key: str, cache: dict[str, Any], profile: dict):
        self.session = requests.Session()
        self.session.auth = (api_key, "")
        self.cache = cache
        self.profile = profile
        self.prefixes = tuple(sorted(profile["sic_sectors"].keys(), key=len, reverse=True))

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                log.warning("Network error (%s), attempt %d/%d", exc, attempt, MAX_RETRIES)
                time.sleep(BACKOFF_BASE_SECONDS * attempt)
                continue
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", BACKOFF_BASE_SECONDS * (2 ** attempt)))
                log.warning("Rate limited (429). Sleeping %.0fs.", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 401:
                log.error("401 Unauthorized — check COMPANIES_HOUSE_API_KEY.")
                sys.exit(1)
            if not resp.ok:
                log.error("CH %s -> %d: %s", url, resp.status_code, resp.text[:150])
                return None
            time.sleep(PER_REQUEST_PAUSE)
            return resp.json()
        return None

    def qualify(self, legal_name: str) -> Optional[dict]:
        """Active, in-scope company -> profile dict (with _sector), else None.

        Caches the raw CH profile (the rate-limited part); the SIC/size filter runs
        fresh each call so filter changes re-apply without re-hitting the API.
        """
        cache_key = f"ch::{legal_name.casefold()}"
        if cache_key in self.cache:
            profile = self.cache[cache_key]
        else:
            profile = self._resolve_profile(legal_name)
            self.cache[cache_key] = profile
        if not profile:
            return None

        sector = self._sector(profile.get("sic_codes", []) or [])
        if not sector:
            return None
        if self.profile["filter_mode"] == "broad":
            acc = ((profile.get("accounts", {}) or {}).get("last_accounts", {}) or {}).get("type", "")
            if acc in MICRO_ACCOUNTS:          # drop the tiniest / dormant shells
                return None

        enriched = dict(profile)
        enriched["_sector"] = sector
        return enriched

    def _resolve_profile(self, legal_name: str) -> Optional[dict]:
        search = self._get(CH_SEARCH_URL, {"q": legal_name, "items_per_page": 5})
        if not search:
            return None
        number = self._best_match_number(legal_name, search.get("items", []))
        if not number:
            return None
        profile = self._get(f"{CH_PROFILE_URL}/{number}")
        if not profile or profile.get("company_status") != "active":
            return None
        return profile

    @staticmethod
    def _best_match_number(legal_name: str, items: list[dict]) -> Optional[str]:
        best, best_score = None, 0.0
        for it in items:
            score = SequenceMatcher(None, legal_name.casefold(), it.get("title", "").casefold()).ratio()
            if score > best_score:
                best, best_score = it, score
        return best.get("company_number") if best and best_score >= NAME_MATCH_THRESHOLD else None

    def _sector(self, sic_codes: list[str]) -> Optional[str]:
        # Match on the PRIMARY (first-listed) SIC only. A company that merely lists
        # a secondary tech/target code as a side activity isn't really in that
        # business, so this removes the "not IT" false positives.
        if not sic_codes:
            return None
        primary = sic_codes[0]
        for prefix in self.prefixes:
            if primary.startswith(prefix):
                return self.profile["sic_sectors"][prefix]
        return None


# ---------------------------------------------------------------------------
# NAME CLEANING + SEARCH LINKS + PROFILE FINDER
# ---------------------------------------------------------------------------

def clean_company_name(name: str) -> str:
    n = name
    parts = re.split(r"\bT/?A\b|\btrading as\b", n, flags=re.I)   # brand after "T/A"
    if len(parts) > 1 and parts[-1].strip():
        n = parts[-1]
    n = LEGAL_SUFFIX_RE.sub(" ", n)
    n = re.sub(r"\(uk\)", " ", n, flags=re.I)
    n = re.sub(r"\bUK\b$", " ", n.strip(), flags=re.I)
    n = re.sub(r"[^\w &.\-]", " ", n)
    n = re.sub(r"\s+", " ", n).strip(" .-&")
    return n or name


def google_search_url(clean: str) -> str:
    return "https://www.google.com/search?q=" + quote(f"{clean} linkedin")


def linkedin_company_url(clean: str) -> str:
    return "https://www.linkedin.com/search/results/companies/?keywords=" + quote(clean)


def clearbit_lookup(brand: str, cache: dict) -> tuple[str, str]:
    """Free, keyless name -> (proper brand name, website) via Clearbit Autocomplete.

    Turns a messy legal name into the name LinkedIn indexes plus the domain, e.g.
    ("Stripe", "https://stripe.com"). Best-effort — the top match can be a namesake,
    so treat the website as a hint. Cached; returns ("", "") on any error.
    """
    key = f"cb::{brand.casefold()}"
    if key in cache:
        c = cache[key]
        return c.get("name", ""), c.get("website", "")
    name, website = "", ""
    try:
        resp = requests.get("https://autocomplete.clearbit.com/v1/companies/suggest",
                            params={"query": brand}, timeout=15)
        if resp.ok and resp.json():
            top = resp.json()[0]
            name = top.get("name", "") or ""
            dom = top.get("domain", "") or ""
            website = f"https://{dom}" if dom else ""
    except Exception:
        pass
    cache[key] = {"name": name, "website": website}
    return name, website


# ---------------------------------------------------------------------------
# STEP 3 — AI DRAFTING
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _enforce_note_limit(note: str) -> str:
    note = note.strip()
    if len(note) <= LINKEDIN_NOTE_MAX:
        return note
    log.warning("Connection note was %d chars; trimming.", len(note))
    window = note[:LINKEDIN_NOTE_MAX]
    end = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    return window[:end + 1].strip() if end >= 120 else re.sub(r"\s+\S*$", "", window).strip()


def draft(client: "anthropic.Anthropic", system_prompt: str,
          company: str, sector: str, town: str) -> tuple[str, str]:
    user_prompt = (f"Company: {company}\nSector: {sector}\nTown: {town or 'UK'}\n\n"
                   "Write the connection note and follow-up DM now.")
    msg = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=ANTHROPIC_MAX_TOKENS,
        system=system_prompt, messages=[{"role": "user", "content": user_prompt}])
    parsed = _parse_json(msg.content[0].text)
    if not parsed:
        return "", msg.content[0].text.strip()
    return _enforce_note_limit(parsed.get("connection_note", "")), parsed.get("followup_dm", "").strip()


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def process_company(ch: "CompaniesHouse", profile: dict, name: str,
                    ai_client: "Optional[anthropic.Anthropic]" = None) -> Optional[dict]:
    """Qualify one register name and return a full lead dict, or None if dropped.

    Shared by the CLI and the Streamlit app so both behave identically.
    """
    prof = ch.qualify(name)
    if not prof:
        return None
    sector = prof["_sector"]
    town = (prof.get("registered_office_address", {}) or {}).get("locality", "")
    number = prof.get("company_number", "")
    lead = Lead(
        company=name, ch_company_name=prof.get("company_name", ""),
        company_number=number, status=prof.get("company_status", ""),
        sic_codes=", ".join(prof.get("sic_codes", []) or []), sector=sector, town=town,
        incorporated=prof.get("date_of_creation", ""),
        target_personas=profile["personas"],
        accounts_type=((prof.get("accounts", {}) or {}).get("last_accounts", {}) or {}).get("type", ""),
        ch_url=f"https://find-and-update.company-information.service.gov.uk/company/{number}",
    )
    clean = clean_company_name(lead.ch_company_name or name)
    cb_name, website = clearbit_lookup(clean, ch.cache)     # proper brand name + website
    lead.linkedin_name = cb_name or clean
    lead.website = website
    lead.google_search = google_search_url(lead.linkedin_name)
    lead.linkedin_company_search = linkedin_company_url(lead.linkedin_name)
    if ai_client:
        try:
            lead.connection_note, lead.followup_dm = draft(
                ai_client, profile["system_prompt"], lead.linkedin_name, sector, town)
            lead.note_chars = len(lead.connection_note)
        except Exception:
            lead.notes = "draft failed"
    return asdict(lead)


def run(args: argparse.Namespace) -> None:
    profile = PROFILES[args.profile]
    out_path = args.out or f"leads_{args.profile}.csv"     # separate file per person
    load_dotenv()
    ch_key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not ch_key:
        sys.exit("COMPANIES_HOUSE_API_KEY is not set. Get a free key at "
                 "https://developer.company-information.service.gov.uk/")

    ai_client = None
    if not args.no_draft:
        if anthropic is None:
            sys.exit("pip install anthropic (or run with --no-draft).")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set (or run with --no-draft).")
        ai_client = anthropic.Anthropic()

    if args.profile == "sister" and "[SISTER_NAME]" in SISTER["name"]:
        log.warning("Edit SISTER['name'] in the script to your sister's real name "
                    "before sending — drafts currently say [SISTER_NAME].")

    cache_path = Path(f".ch_cache_{args.profile}.json")     # separate cache per profile
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ch = CompaniesHouse(ch_key, cache, profile)

    companies = load_sponsors(args.csv, args.require_a_rating)
    companies = companies[args.skip: args.skip + args.limit]
    log.info("Profile '%s' | processing register rows %d-%d (%d companies).",
             args.profile, args.skip, args.skip + len(companies), len(companies))

    leads: list[dict] = []
    try:
        for i, name in enumerate(companies, 1):
            d = process_company(ch, profile, name, ai_client)
            if not d:
                continue
            log.info("[%d/%d] KEEP %s — %s (%s)", i, len(companies),
                     d["linkedin_name"] or d["ch_company_name"], d["sector"], d["town"])
            leads.append(d)
    finally:
        cache_path.write_text(json.dumps(ch.cache))

    if leads:
        out = pd.DataFrame(leads)
        out["_p"] = out["sector"].map(profile["sector_priority"]).fillna(9)
        out = out.sort_values(["_p", "ch_company_name"]).drop(columns="_p")
        cols = ["company", "linkedin_name", "website", "target_personas", "sector",
                "town", "accounts_type", "linkedin_company_search", "google_search",
                "ch_url", "company_number", "sic_codes", "incorporated", "status",
                "connection_note", "note_chars", "followup_dm", "notes"]
        out[[c for c in cols if c in out.columns]].to_csv(out_path, index=False)
    log.info("Done: %d scanned -> %d qualified -> %s", len(companies), len(leads), out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Free LinkedIn-first job-search engine.")
    p.add_argument("--csv", required=True, help="Path to the gov.uk sponsor register CSV.")
    p.add_argument("--profile", choices=list(PROFILES), default="nandu",
                   help="Which candidate profile / filter to use.")
    p.add_argument("--out", default=None, help="Output CSV path (default: leads_<profile>.csv).")
    p.add_argument("--limit", type=int, default=200, help="How many register rows to process.")
    p.add_argument("--skip", type=int, default=0, help="Offset — page through the register across runs.")
    p.add_argument("--require-a-rating", action="store_true",
                   help="Keep only A-rated sponsors (B-rated can't issue new CoS).")
    p.add_argument("--no-draft", action="store_true",
                   help="Produce the qualified-company list only; skip AI drafting.")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
