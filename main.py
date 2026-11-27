"""
Campus Lost and Found Matcher  (Gemini edition)

Division of work (as per the spec):
  * Gemini LLM  -> reads each report and extracts item, colour, brand, place, summary (JSON)
  * Python      -> validates the LLM output, applies the matching rules, scores pairs,
                   and writes matches.csv / no_match.csv / unclaimed.csv / summary.txt

The system only SUGGESTS matches. It never decides ownership.

Usage:
    python main.py            # uses Gemini (needs GEMINI_API_KEY in .env)
    python main.py --offline  # rule-based stand-in extractor, only for testing the pipeline
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from itertools import groupby
from pathlib import Path

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output"

MAX_DAYS_APART = 30
MIN_REPORTS = 10
SCORE_COLOUR, SCORE_BRAND, SCORE_PLACE = 2, 2, 1
LIKELY_MIN = 3

HELPDESK_NOTE = ("Do NOT hand over yet. Ask the claimant for an extra identifying detail "
                 "that appears in neither report.")

# ---------------------------------------------------------------- categories
CATEGORY_KEYWORDS = {
    "phone": ["phone", "mobile", "smartphone", "iphone"],
    "wallet": ["wallet", "purse"],
    "bottle": ["bottle", "flask", "tumbler"],
    "earphones": ["earphone", "earbud", "headphone", "airpod", "headset"],
    "umbrella": ["umbrella"],
    "watch": ["watch"],
    "bag": ["bag", "backpack", "rucksack", "satchel"],
    "calculator": ["calculator"],
    "keys": ["key", "keys", "keychain"],
    "laptop": ["laptop", "notebook computer"],
    "id card": ["id card", "identity card", "student card"],
}


def categorise(item):
    """Map the LLM's item text to a fixed category. None = vague, 'other' = unknown type."""
    if not item:
        return None
    text = item.lower()
    for cat, words in CATEGORY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(w)}s?\b", text) for w in words):
            return cat
    return "other"


# ---------------------------------------------------------------- input
def read_reports(path, label):
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"id", "date", "report"} <= set(rows[0].keys()):
        sys.exit(f"ERROR: {path.name} must have columns: id, date, report")
    reports = []
    for i, r in enumerate(rows, start=2):
        rid, date, text = (r.get("id") or "").strip(), (r.get("date") or "").strip(), (r.get("report") or "").strip()
        if not rid or not date or not text:
            sys.exit(f"ERROR: {path.name} line {i}: id, date and report are all required.")
        try:
            d = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            sys.exit(f"ERROR: {path.name} line {i}: date '{date}' must be YYYY-MM-DD.")
        reports.append({"id": rid, "date": d, "text": text})
    if len(reports) < MIN_REPORTS:
        sys.exit(f"ERROR: {path.name} has {len(reports)} reports; at least {MIN_REPORTS} are required.")
    return reports


# ---------------------------------------------------------------- extraction
PROMPT = """You extract details from a campus lost-and-found report.

Return ONLY a JSON object with exactly these keys:
  "item":    the type of item as a short noun (e.g. "phone", "wallet", "backpack"), or null if no item type is stated
  "colour":  a colour word that appears in the report, or null
  "brand":   a brand name that appears in the report, or null
  "place":   the location noun phrase from the report (e.g. "library", "canteen", "mathematics classroom"), or null
  "summary": a summary of 8 words or fewer

Strict rules:
- colour, brand and place must literally appear in the report text. If they do not, use null.
- NEVER guess or infer a brand (e.g. do not assume Samsung or Apple from the word "phone").
- If the report is vague (e.g. "I lost something near the library"), use null for every missing field. Do not invent details.
- No explanations, no markdown, JSON only.

Report: \"\"\"{text}\"\"\""""


def call_gemini(client, model, text, retries=4):
    from google.genai import types
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
    last = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=PROMPT.format(text=text), config=config)
            raw = (resp.text or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            data = json.loads(raw)
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                raise ValueError("response was not a JSON object")
            return data
        except Exception as e:  # network, quota, bad JSON ...
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini failed after {retries} attempts: {last}")


COLOURS = ["black", "white", "red", "blue", "green", "yellow", "brown", "silver", "grey", "gray",
           "pink", "orange", "purple", "gold", "golden", "navy", "maroon", "beige", "transparent"]
BRANDS = ["samsung", "apple", "boat", "casio", "sony", "nike", "adidas", "puma", "jbl", "oneplus",
          "realme", "xiaomi", "dell", "hp", "lenovo", "titan", "fastrack", "fossil", "skybags", "wildcraft"]


def offline_extract(text):
    """Simple rule-based stand-in so the pipeline can be tested without an API key."""
    t = text.lower()
    item = None
    for cat, words in CATEGORY_KEYWORDS.items():
        for w in words:
            if re.search(rf"\b{w}s?\b", t):
                item = w
                break
        if item:
            break
    colour = next((c for c in COLOURS if re.search(rf"\b{c}\b", t)), None)
    brand = next((b for b in BRANDS if re.search(rf"\b{b}\b", t)), None)
    m = re.search(r"\b(?:near|in|at|from|outside|on)\s+(?:the\s+)?([a-z ]+?)(?:\.|$| table| bench)", t)
    place = m.group(1).strip() if m else None
    return {"item": item, "colour": colour, "brand": brand, "place": place,
            "summary": " ".join(text.split()[:8])}


def clean(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in ("", "null", "none", "n/a", "unknown") else value


def validate_extraction(raw, text):
    """Python-side safety net: values must literally appear in the text, summary <= 8 words."""
    low = text.lower()
    out = {k: clean(raw.get(k)) for k in ("item", "colour", "brand", "place", "summary")}
    for field in ("colour", "brand"):
        v = out[field]
        if v and v.lower() not in low:
            out[field] = None
    if out["place"]:
        words = re.findall(r"[a-z]+", out["place"].lower())
        if not words or not all(w in low for w in words):
            out["place"] = None
    if out["item"] and out["item"].lower() not in low:
        # allow plural/singular variants, otherwise treat the item as not stated
        stem = out["item"].lower().rstrip("s")
        if stem not in low:
            out["item"] = None
    summary = out["summary"] or ""
    words = summary.split()
    out["summary"] = " ".join(words[:8]) if words else " ".join(text.split()[:8])
    return out


def extract_all(reports, kind, client, model, offline):
    results = []
    for r in reports:
        raw = offline_extract(r["text"]) if offline else call_gemini(client, model, r["text"])
        ex = validate_extraction(raw, r["text"])
        category = categorise(ex["item"])
        needs_review = category is None or (ex["colour"] is None and ex["brand"] is None and ex["place"] is None)
        results.append({**r, **ex, "category": category, "manual_review": needs_review})
        print(f"  [{kind}] {r['id']}: {json.dumps({k: ex[k] for k in ('item','colour','brand','place','summary')})}")
    return results


# ---------------------------------------------------------------- matching
def norm_place(p):
    return re.sub(r"\b(the|a|an)\b", "", p.lower()).split() if p else []


def same_place(a, b):
    wa, wb = norm_place(a), norm_place(b)
    if not wa or not wb:
        return False
    return wa == wb or wa[-1] == wb[-1]  # "mathematics classroom" ~ "classroom"


def compare(lost, found):
    """Return (match_dict | None, rejection_reason | None)."""
    if lost["category"] is None or found["category"] is None:
        return None, "vague report"
    if lost["category"] == "other" or found["category"] == "other":
        return None, "'other' category never auto-matched"
    if lost["category"] != found["category"]:
        return None, "different item category"
    gap = abs((lost["date"] - found["date"]).days)
    if gap > MAX_DAYS_APART:
        return None, f"dates {gap} days apart"
    if lost["colour"] and found["colour"] and lost["colour"].lower() != found["colour"].lower():
        return None, "colour conflict"
    if lost["brand"] and found["brand"] and lost["brand"].lower() != found["brand"].lower():
        return None, "brand conflict"

    score, reasons = 0, [f"same item category ({lost['category']})"]
    if lost["colour"] and found["colour"]:
        score += SCORE_COLOUR
        reasons.append(f"same colour ({found['colour']}) +{SCORE_COLOUR}")
    if lost["brand"] and found["brand"]:
        score += SCORE_BRAND
        reasons.append(f"same brand ({found['brand']}) +{SCORE_BRAND}")
    if same_place(lost["place"], found["place"]):
        score += SCORE_PLACE
        reasons.append(f"same place ({lost['place']} / {found['place']}) +{SCORE_PLACE}")
    reasons.append(f"reported {gap} day(s) apart")
    if score == 0:
        return None, "score 0"
    return {"lost_id": lost["id"], "found_id": found["id"], "score": score,
            "category": "likely" if score >= LIKELY_MIN else "possible",
            "reasons": "; ".join(reasons), "gap": gap}, None


def run_matching(lost, found):
    matches = []
    for l in lost:
        for f in found:
            m, _ = compare(l, f)
            if m:
                matches.append(m)
    matches.sort(key=lambda m: (-m["score"], m["gap"], m["lost_id"], m["found_id"]))
    # double claims: one found item suggested for >1 lost report
    by_found = {}
    for m in matches:
        by_found.setdefault(m["found_id"], []).append(m)
    double = {fid: ms for fid, ms in by_found.items() if len(ms) > 1}
    for i, m in enumerate(matches, start=1):
        m["rank"] = i
        claim = m["found_id"] in double
        m["double_claim"] = "YES - " + ", ".join(x["lost_id"] for x in double[m["found_id"]]) if claim else "no"
        m["helpdesk_action"] = HELPDESK_NOTE
    return matches, double


# ---------------------------------------------------------------- output
def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_outputs(lost, found, matches, double):
    OUTPUT_DIR.mkdir(exist_ok=True)
    lmap = {r["id"]: r for r in lost}
    fmap = {r["id"]: r for r in found}

    rows = []
    for m in matches:
        l, f = lmap[m["lost_id"]], fmap[m["found_id"]]
        rows.append({**m, "lost_report": l["text"], "found_report": f["text"]})
    write_csv(OUTPUT_DIR / "matches.csv",
              ["rank", "lost_id", "found_id", "score", "category", "reasons", "double_claim",
               "lost_report", "found_report", "helpdesk_action"], rows)

    matched_lost = {m["lost_id"] for m in matches}
    matched_found = {m["found_id"] for m in matches}

    no_match = []
    for l in lost:
        if l["id"] not in matched_lost:
            if l["manual_review"]:
                reason = "MANUAL REVIEW - report is vague or incomplete (missing item/details); no automatic match attempted"
            elif l["category"] == "other":
                reason = "MANUAL REVIEW - 'other' category items are never matched automatically"
            else:
                reason = "No found report passed the matching rules"
            no_match.append({"id": l["id"], "date": l["date"], "report": l["text"],
                             "item": l["item"], "colour": l["colour"], "brand": l["brand"], "place": l["place"],
                             "manual_review": "YES" if (l["manual_review"] or l["category"] == "other") else "no",
                             "reason": reason})
    write_csv(OUTPUT_DIR / "no_match.csv",
              ["id", "date", "report", "item", "colour", "brand", "place", "manual_review", "reason"], no_match)

    unclaimed = [{"id": f["id"], "date": f["date"], "report": f["text"], "item": f["item"],
                  "colour": f["colour"], "brand": f["brand"], "place": f["place"]}
                 for f in found if f["id"] not in matched_found]
    write_csv(OUTPUT_DIR / "unclaimed.csv",
              ["id", "date", "report", "item", "colour", "brand", "place"], unclaimed)

    likely = sum(m["category"] == "likely" for m in matches)
    possible = len(matches) - likely
    review = [l for l in lost if l["manual_review"] or l["category"] == "other"]
    lines = ["LOST AND FOUND MATCHER - SUMMARY", "=" * 40,
             f"Run time              : {datetime.now():%Y-%m-%d %H:%M}",
             f"Lost reports          : {len(lost)}",
             f"Found reports         : {len(found)}",
             f"Suggested pairs       : {len(matches)} (likely: {likely}, possible: {possible})",
             f"Lost with no match    : {len(no_match)}",
             f"Unclaimed found items : {len(unclaimed)}",
             f"Manual review flags   : {len(review)}  {[r['id'] for r in review]}",
             f"Double claims         : {len(double)}", ""]
    if double:
        lines.append("DOUBLE CLAIM ALERTS")
        lines.append("-" * 40)
        for fid, ms in double.items():
            lines.append(f"Found item {fid}: \"{fmap[fid]['text']}\"")
            for m in ms:
                lines.append(f"   - {m['lost_id']} | score {m['score']} ({m['category']}) | {m['reasons']}")
            lines.append(f"   ACTION: {HELPDESK_NOTE} Ask EVERY claimant listed above.")
            lines.append("")
    lines.append("NOTE: These are suggestions only. The help desk must confirm the owner.")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="test mode without Gemini (rule-based extractor)")
    args = ap.parse_args()

    client = model = None
    if not args.offline:
        try:
            from dotenv import load_dotenv
            load_dotenv(BASE / ".env")
        except ImportError:
            sys.exit("Run: pip install -r requirements.txt")
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key or key == "PASTE_YOUR_API_KEY_HERE":
            sys.exit("ERROR: put your key in the .env file: GEMINI_API_KEY=your_key")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        try:
            from google import genai
        except ImportError:
            sys.exit("Run: pip install -r requirements.txt")
        client = genai.Client(api_key=key)
        print(f"Using Gemini model: {model}")
    else:
        print("OFFLINE test mode (no LLM) - results are for pipeline testing only.")

    lost = read_reports(INPUT_DIR / "lost.csv", "lost")
    found = read_reports(INPUT_DIR / "found.csv", "found")

    print("\nExtracting lost reports...")
    lost = extract_all(lost, "lost", client, model, args.offline)
    print("\nExtracting found reports...")
    found = extract_all(found, "found", client, model, args.offline)

    matches, double = run_matching(lost, found)
    summary = write_outputs(lost, found, matches, double)
    print("\n" + summary)
    print(f"\nFiles written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
