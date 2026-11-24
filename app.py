"""Medicine Mirror: check a medicine strip from a photo.

Flow: photo -> Gemini reads the printed text and where it sits -> plain code
verifies expiry and the CDSCO flagged-batch list -> verdict, highlights, voice.
"""
import base64
import datetime
import hashlib
import html
import io
import json
import os
import re
import time

import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont, ImageOps

st.set_page_config(page_title="Medicine Mirror", page_icon="💊", layout="centered")

# ----------------------------------------------------------------- settings
OPENROUTER_MODELS = ["heygen/heygen-voice-1-turbo:free","anthropic/claude-sonnet-5.5:batch","nvidia/nemotron-3-ultra-550b-a55b:free","perplexity/pplx-decider-v1.3:free","gemini-4-hydrogen", "gemini-3.8-flash","recraft/recraft-v4.1-flash-v2:free","google/gemma-4-31b-it:free","fish-audio/s2.1-pro-free:free"]   # tried first; put your chosen free vision models before this
MODELS = ["gemini-4-hydrogen", "gemini-3.8-flash"]   # Gemini backup, tried in this order
TRUST_OPENROUTER_BOXES = False   # highlight boxes from OpenRouter models are not drawn unless you set True
MAX_PHOTOS = 4          # photos per scan
SESSION_LIMIT = 8       # scans per visitor session
DAILY_LIMIT = 100       # photos per day for the whole site (protects the free quotas)
DEVELOPER_NAME = "SRIRAM NAGESH"                                  # <- put your name here
LINKEDIN_URL = "https://www.linkedin.com/in/sriram-nagesh-yellaboena/"    # <- put your LinkedIn link here

PROMPT = """You are reading a photo of a medicine strip, blister pack or box.
Read everything printed on it and reply with ONLY a JSON object, no other text, using these keys:
brand_name, composition, batch_no, mfg_date, expiry_date, manufacturer, marketed_by,
mfg_licence_no, mrp, visible_problems, spelling_mistakes, reading_confidence, boxes.
Rules:
- Copy text exactly as printed. Use null for anything you cannot see. Never guess.
- Dates must be written exactly as printed on the strip.
- visible_problems is a list of short strings describing anything suspicious you can
  actually see (misspelled words, smudged or uneven printing, mismatched fonts,
  contradictory details). Use an empty list if you see nothing.
- reading_confidence is "high", "medium" or "low" for how clearly the photo could be read.
- spelling_mistakes is a list of objects like {"printed": "...", "expected": "..."} for words on the
  strip that are clearly misspelled, for example a brand or ingredient name with a wrong letter.
  Report a word only when the text is clearly legible and you are sure it is wrong. Never report text
  that is blurred, cut off, or hidden by glare or the foil pattern. Use an empty list if there are none.
- boxes is an object. For every field you could read (all keys above except
  visible_problems, spelling_mistakes, reading_confidence and boxes), give where its text sits in the photo as
  [ymin, xmin, ymax, xmax], each a number from 0 to 1000 (position as a fraction of the
  image height or width, times 1000). Leave out fields you could not read."""

LABELS = {"brand_name": "Brand", "composition": "Composition", "batch_no": "Batch",
          "mfg_date": "Mfg date", "expiry_date": "Expiry", "manufacturer": "Manufacturer",
          "marketed_by": "Marketed by", "mfg_licence_no": "Licence no.", "mrp": "MRP"}
COLORS = {"ok": "#1B7F4B", "warn": "#B26A00", "bad": "#C62828"}
RANK = {"ok": 0, "warn": 1, "bad": 2}
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}

# ------------------------------------------------------------------- design
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Noto+Sans+Devanagari:wght@400;700&family=Noto+Sans+Telugu:wght@400;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp button, .stApp input, .stApp [data-testid="stCaptionContainer"] {
  font-family: 'Atkinson Hyperlegible', 'Noto Sans Telugu', 'Noto Sans Devanagari', sans-serif;
}
.block-container { max-width: 760px; padding-top: 2.2rem; }
.mm-title { font-size: 2.5rem; font-weight: 700; color: #14213D; letter-spacing: -0.01em; margin: 0; }
.mm-sub { font-size: 1.15rem; color: #3B4A63; margin: 0.2rem 0 1.3rem; }
.mm-points { margin: 0 0 1rem; padding-left: 1.2rem; color: #14213D; font-size: 1.05rem; }
.mm-points li { margin: 0.25rem 0; }
.verdict { border-radius: 14px; padding: 1.1rem 1.3rem; border: 2px solid; margin: 0.6rem 0 1rem; }
.verdict .label { font-size: 1.75rem; font-weight: 700; line-height: 1.2; }
.verdict ul { margin: 0.5rem 0 0; padding-left: 1.2rem; font-size: 1.05rem; }
.verdict.bad { background: #FDECEA; border-color: #C62828; color: #7A1111; }
.verdict.caution { background: #FFF4E0; border-color: #B26A00; color: #5E3800; }
.verdict.ok { background: #E8F5EE; border-color: #1B7F4B; color: #0F4A2B; }
.kv { display: grid; grid-template-columns: 9.5rem 1fr; gap: 0.4rem 1rem; background: #FFFFFF;
      border: 1px solid #D5DBE3; border-radius: 12px; padding: 0.9rem 1.1rem; }
.kv .k { color: #51607A; }
.kv .v { font-weight: 700; color: #14213D; overflow-wrap: anywhere; }
.kv .v.none { font-weight: 400; color: #7A869B; }
.checks { list-style: none; padding: 0; margin: 0; }
.checks li { padding: 0.5rem 0; border-bottom: 1px solid #E1E6EC; display: flex; gap: 0.6rem; color: #14213D; }
.mm-devwrap { text-align: center; padding: 0.4rem 0 1.5rem; }
.mm-dev { display: inline-flex; align-items: center; min-height: 44px; padding: 0 0.6rem; font-weight: 700;
          color: #2447C5 !important; text-decoration: underline; }
.mm-foot { font-size: 0.92rem; color: #51607A; border-top: 1px solid #D5DBE3; margin-top: 2rem; padding-top: 1rem; }
@media (max-width: 600px) {
  .kv { grid-template-columns: 1fr; }
  .mm-title { font-size: 2rem; }
}
</style>
"""

# -------------------------------------------------------------------- voice
LANGS = {"English": "en", "Telugu (తెలుగు)": "te", "Hindi (हिन्दी)": "hi"}

MONTH_NAMES = {
    "en": ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
           "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    "te": ["జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై",
           "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"],
}

SCRIPTS = {
    "en": {
        "intro": "Here is what I found on your medicine strip.",
        "brand": "The medicine is {brand}.",
        "composition": "It contains {composition}.",
        "batch": "The batch number is {batch}.",
        "manufacturer": "It is made by {manufacturer}.",
        "manufacturer_missing": "The manufacturer name is not visible in this photo. You can add another photo that shows it.",
        "expiry_ok": "The expiry date is {month} {year}. The medicine is still within its date.",
        "expiry_soon": "The expiry date is {month} {year}, so it will expire soon. Please check with a pharmacist before using it.",
        "expired": "This medicine has expired. Do not use it.",
        "expiry_unknown": "The expiry date could not be read clearly.",
        "batch_flagged": "Warning. This batch was listed by the drug regulator as not of standard quality. Do not use it, and show it to a pharmacist or doctor.",
        "batch_ok": "This batch was not found in the flagged lists I checked. This does not prove that the medicine is genuine.",
        "batch_unknown": "The batch number could not be read clearly.",
        "photo_caution": "The photo shows some signs that need caution.",
        "spelling_bad": "The printing shows possible spelling mistakes. Do not use this medicine, and show it to a pharmacist.",
        "final_bad": "Overall result: do not use this medicine. Please show it to a pharmacist.",
        "final_caution": "Overall result: be careful. Some details could not be confirmed.",
        "final_ok": "Overall result: no warning signs found. This is only a screening check, not proof that the medicine is genuine.",
    },
    "hi": {
        "intro": "आपकी दवा की स्ट्रिप पर मुझे यह जानकारी मिली।",
        "brand": "दवा का नाम {brand} है।",
        "composition": "इसमें {composition} है।",
        "batch": "बैच नंबर {batch} है।",
        "manufacturer": "इसे {manufacturer} ने बनाया है।",
        "manufacturer_missing": "इस फ़ोटो में बनाने वाली कंपनी का नाम दिखाई नहीं दे रहा। आप उसे दिखाने वाली एक और फ़ोटो जोड़ सकते हैं।",
        "expiry_ok": "एक्सपायरी की तारीख {month} {year} है। दवा अभी तारीख के अंदर है।",
        "expiry_soon": "एक्सपायरी की तारीख {month} {year} है, यानी यह जल्दी ख़त्म होने वाली है। इस्तेमाल से पहले फार्मासिस्ट से पूछ लें।",
        "expired": "यह दवा एक्सपायर हो चुकी है। इसे इस्तेमाल न करें।",
        "expiry_unknown": "एक्सपायरी की तारीख साफ़ नहीं पढ़ी जा सकी।",
        "batch_flagged": "चेतावनी। इस बैच को दवा नियामक ने गुणवत्ता मानक पर खरा न उतरने वाला बताया है। इसे इस्तेमाल न करें और किसी फार्मासिस्ट या डॉक्टर को दिखाएँ।",
        "batch_ok": "यह बैच मेरी जाँची हुई चिह्नित सूचियों में नहीं मिला। इसका मतलब यह नहीं कि दवा असली है।",
        "batch_unknown": "बैच नंबर साफ़ नहीं पढ़ा जा सका।",
        "photo_caution": "फ़ोटो में कुछ ऐसी बातें दिखीं जिनके लिए सावधानी ज़रूरी है।",
        "spelling_bad": "छपाई में वर्तनी की संभावित ग़लतियाँ दिखीं। इस दवा का इस्तेमाल न करें और किसी फार्मासिस्ट को दिखाएँ।",
        "final_bad": "कुल नतीजा: इस दवा का इस्तेमाल न करें। कृपया इसे किसी फार्मासिस्ट को दिखाएँ।",
        "final_caution": "कुल नतीजा: सावधान रहें। कुछ जानकारी की पुष्टि नहीं हो सकी।",
        "final_ok": "कुल नतीजा: ख़तरे का कोई संकेत नहीं मिला। यह सिर्फ़ एक शुरुआती जाँच है, दवा के असली होने का सबूत नहीं।",
    },
    "te": {
        "intro": "మీ మందు స్ట్రిప్ మీద నాకు ఈ సమాచారం కనిపించింది.",
        "brand": "మందు పేరు {brand}.",
        "composition": "ఇందులో {composition} ఉంది.",
        "batch": "బ్యాచ్ నంబర్ {batch}.",
        "manufacturer": "దీన్ని {manufacturer} తయారు చేసింది.",
        "manufacturer_missing": "ఈ ఫోటోలో తయారీదారు పేరు కనిపించలేదు. అది కనిపించే మరో ఫోటో జోడించవచ్చు.",
        "expiry_ok": "గడువు తేదీ {month} {year}. మందు ఇంకా గడువు లోపలే ఉంది.",
        "expiry_soon": "గడువు తేదీ {month} {year}, అంటే త్వరలో ముగుస్తుంది. వాడే ముందు ఫార్మసిస్ట్‌ను అడగండి.",
        "expired": "ఈ మందు గడువు ముగిసింది. దీన్ని వాడకండి.",
        "expiry_unknown": "గడువు తేదీని స్పష్టంగా చదవడం సాధ్యం కాలేదు.",
        "batch_flagged": "హెచ్చరిక. ఈ బ్యాచ్‌ను ఔషధ నియంత్రణ సంస్థ నాణ్యత ప్రమాణాలకు తగదని ప్రకటించింది. దీన్ని వాడకండి, ఫార్మసిస్ట్‌కు లేదా డాక్టర్‌కు చూపించండి.",
        "batch_ok": "ఈ బ్యాచ్ నేను పరిశీలించిన హెచ్చరిక జాబితాల్లో కనిపించలేదు. అంటే మందు అసలైనదని కాదు.",
        "batch_unknown": "బ్యాచ్ నంబర్‌ను స్పష్టంగా చదవడం సాధ్యం కాలేదు.",
        "photo_caution": "ఫోటోలో కొన్ని జాగ్రత్త అవసరమయ్యే సూచనలు కనిపించాయి.",
        "spelling_bad": "ప్రింటింగ్‌లో అక్షర దోషాలు ఉండవచ్చని కనిపించింది. ఈ మందును వాడకండి, ఫార్మసిస్ట్‌కు చూపించండి.",
        "final_bad": "మొత్తం ఫలితం: ఈ మందును వాడకండి. దయచేసి ఫార్మసిస్ట్‌కు చూపించండి.",
        "final_caution": "మొత్తం ఫలితం: జాగ్రత్తగా ఉండండి. కొన్ని వివరాలను నిర్ధారించలేకపోయాము.",
        "final_ok": "మొత్తం ఫలితం: ప్రమాద సూచనలు ఏవీ కనిపించలేదు. ఇది కేవలం ప్రాథమిక పరిశీలన మాత్రమే, మందు అసలైనదని రుజువు కాదు.",
    },
}


# --------------------------------------------------------------- data + AI
def get_secret(name):
    try:
        value = st.secrets[name]
        if value and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(name) or "").strip() or None


def get_api_key():
    return get_secret("OPENROUTER_API_KEY") or get_secret("GEMINI_API_KEY")


@st.cache_data
def load_data():
    df = pd.read_csv("data/nsq_flagged.csv", dtype=str).fillna("")
    df["batch_key"] = df["batch_no"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return df


def image_to_bytes(image, max_side=1600):
    img = image.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON in the reply")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("unexpected AI output")
    return data


def read_with_gemini(img_bytes):
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no backup key set")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=40000))
    last_error = None
    for model in MODELS:
        for attempt in range(2):
            began = time.time()
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), PROMPT],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                data = extract_json(resp.text)
                print(f"[gemini {model}] OK in {time.time() - began:.1f}s", flush=True)
                return data
            except Exception as e:
                last_error = e
                print(f"[gemini {model}] FAILED after {time.time() - began:.1f}s: {str(e)[:120]}", flush=True)
                if any(code in str(e) for code in ("503", "UNAVAILABLE", "429")):
                    time.sleep(1)
                    continue
                break
    raise last_error


@st.cache_data(show_spinner=False, ttl=3600)
def read_with_ai(img_bytes):
    """OpenRouter free models first; Gemini only if every OpenRouter try fails."""
    last_error = None
    key = get_secret("OPENROUTER_API_KEY")
    if key:
        b64 = base64.b64encode(img_bytes).decode()
        for model in OPENROUTER_MODELS:
            began = time.time()
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "X-Title": "Medicine Mirror"},
                    json={"model": model, "max_tokens": 1500,
                          "messages": [{"role": "user", "content": [
                              {"type": "text", "text": PROMPT},
                              {"type": "image_url",
                               "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]},
                    timeout=40)
                r.raise_for_status()
                body = r.json()
                data = extract_json(body["choices"][0]["message"]["content"])
                data["_source"] = "openrouter"
                data["_model"] = body.get("model", model)
                print(f"[openrouter {model}] OK in {time.time() - began:.1f}s", flush=True)
                return data
            except Exception as e:
                last_error = e
                detail = getattr(getattr(e, "response", None), "text", "") or str(e)
                print(f"[openrouter {model}] FAILED after {time.time() - began:.1f}s: {detail[:150]}", flush=True)
    if get_secret("GEMINI_API_KEY"):
        data = read_with_gemini(img_bytes)
        data["_source"] = "gemini"
        return data
    raise last_error or RuntimeError("no reading service is set up")


@st.cache_resource
def usage_counter():
    return {"day": datetime.date.today(), "count": 0}


def allow_scan(sig):
    """Protect the free quota: limit scans per visitor and per day."""
    if st.session_state.get("last_sig") == sig:
        return True                                  # same photos, page rerun
    counter = usage_counter()
    today = datetime.date.today()
    if counter["day"] != today:
        counter["day"], counter["count"] = today, 0
    if st.session_state.get("scans", 0) >= SESSION_LIMIT or counter["count"] >= DAILY_LIMIT:
        return False
    st.session_state["scans"] = st.session_state.get("scans", 0) + 1
    counter["count"] += len(sig)
    st.session_state["last_sig"] = sig
    return True


# ------------------------------------------------------------------- checks
def is_missing(v):
    return v is None or str(v).strip().lower() in ("", "null", "none", "not visible", "n/a")


def parse_month_year(text):
    t = str(text or "").upper().replace(".", "/").replace("-", "/")
    m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})", t)   # 15/02/2028
    if m:
        month, year_txt = int(m.group(2)), m.group(3)
    else:
        m = re.search(r"(\d{1,2})\s*/\s*(\d{2,4})", t)                # 02/2028
        if m:
            month, year_txt = int(m.group(1)), m.group(2)
        else:
            m = re.search(r"([A-Z]{3})[A-Z]*\s*/?\s*([0-9OIS]{2,4})", t)   # FEB.2028
            if not m or m.group(1) not in MONTHS:
                return None
            month, year_txt = MONTHS[m.group(1)], m.group(2)
    year = int(year_txt.translate(str.maketrans("OIS", "015")))
    if year < 100:
        year += 2000
    if not (1 <= month <= 12 and 2018 <= year <= 2045):
        return None
    return (year, month)


def clean_box(b):
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in b]
    except Exception:
        return None
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        return None
    return (ymin, xmin, ymax, xmax)


def merge_reads(reads):
    """Combine several photos: the first clear value for each field wins."""
    merged, boxes = {}, {}
    for key in LABELS:
        for i, d in enumerate(reads):
            if not is_missing(d.get(key)):
                merged[key] = d[key]
                raw = d.get("boxes") if isinstance(d.get("boxes"), dict) else {}
                box = clean_box(raw.get(key))
                if box:
                    boxes[key] = (i, box)
                break
    problems = []
    for d in reads:
        for p in d.get("visible_problems") or []:
            if p not in problems:
                problems.append(p)
    merged["visible_problems"] = problems
    spelling = []
    for d in reads:
        for item in d.get("spelling_mistakes") or []:
            if item not in spelling:
                spelling.append(item)
    merged["spelling_mistakes"] = spelling
    order = {"low": 0, "medium": 1, "high": 2}
    confs = [d.get("reading_confidence") for d in reads if d.get("reading_confidence") in order]
    merged["reading_confidence"] = min(confs, key=order.get) if confs else None
    return merged, boxes


def find_conflicts(reads):
    msgs = []
    batches = {re.sub(r"[^A-Z0-9]", "", str(d["batch_no"]).upper())
               for d in reads if not is_missing(d.get("batch_no"))}
    if len(batches) > 1:
        msgs.append("The photos show different batch numbers. Make sure they are of the same strip.")
    expiries = {parse_month_year(d["expiry_date"])
                for d in reads if not is_missing(d.get("expiry_date"))} - {None}
    if len(expiries) > 1:
        msgs.append("The photos show different expiry dates. Make sure they are of the same strip.")
    return msgs


def spelling_text(item):
    if isinstance(item, dict):
        printed, expected = item.get("printed"), item.get("expected")
        if printed and expected:
            return f"'{printed}' (should probably be '{expected}')"
        return str(printed or expected or "")
    return str(item)


def check_strip(d, df):
    checks, levels = [], {}

    def add(level, msg, field=None):
        checks.append((level, msg))
        if field and RANK[level] >= RANK[levels.get(field, "ok")]:
            levels[field] = level

    today = datetime.date.today()
    exp, mfg = parse_month_year(d.get("expiry_date")), parse_month_year(d.get("mfg_date"))

    if exp:
        left = (exp[0] - today.year) * 12 + (exp[1] - today.month)
        if left < 0:
            add("bad", f"Expired: the expiry date was {exp[1]:02d}/{exp[0]}.", "expiry_date")
        elif left <= 3:
            add("warn", f"Expires soon: {exp[1]:02d}/{exp[0]} ({left} month(s) left).", "expiry_date")
        else:
            add("ok", f"Expiry {exp[1]:02d}/{exp[0]} is valid ({left} months left).", "expiry_date")
    else:
        add("warn", "The expiry date could not be read or understood.", "expiry_date")

    if exp and mfg:
        shelf = (exp[0] - mfg[0]) * 12 + (exp[1] - mfg[1])
        if shelf <= 0:
            add("warn", "The manufacturing date is not before the expiry date. Check both dates on the strip.", "mfg_date")
        elif not 6 <= shelf <= 60:
            add("warn", f"Unusual shelf life: {shelf} months between manufacture and expiry.", "mfg_date")
        else:
            add("ok", f"Shelf life of {shelf} months looks plausible.", "mfg_date")
    elif not mfg:
        add("warn", "The manufacturing date could not be read clearly.", "mfg_date")

    if d.get("batch_no"):
        key = re.sub(r"[^A-Z0-9]", "", str(d["batch_no"]).upper())
        hit = df[df["batch_key"] == key]
        if not hit.empty:
            r = hit.iloc[0]
            add("bad", f"Batch {r['batch_no']} was flagged by the drug regulator as Not of "
                       f"Standard Quality ({r['month']}): {r['reason']}.", "batch_no")
        else:
            add("ok", f"Batch {d['batch_no']} is not in the flagged lists we checked. "
                      "This alone does not prove it is genuine.", "batch_no")
    else:
        add("warn", "The batch number could not be read.", "batch_no")

    if d.get("manufacturer"):
        add("ok", f"Manufacturer printed on the strip: {d['manufacturer']}.", "manufacturer")
    else:
        add("warn", "The manufacturer name is not visible in this photo.")

    if is_missing(d.get("brand_name")):
        add("warn", "The brand name could not be read clearly.", "brand_name")
    if is_missing(d.get("composition")):
        add("warn", "The composition could not be read clearly.", "composition")
    hard_photo = d.get("reading_confidence") == "low"
    for item in d.get("spelling_mistakes") or []:
        if hard_photo:
            add("warn", f"A possible spelling mistake was seen ({spelling_text(item)}), but the photo is "
                        "hard to read. Retake it flat, in even light.")
        else:
            add("bad", f"Possible spelling mistake in the printing: {spelling_text(item)}. "
                       "Genuine strips are printed carefully, so show this one to a pharmacist.")
    for problem in d.get("visible_problems") or []:
        add("warn", f"The AI noticed: {problem}")
    if d.get("reading_confidence") == "low":
        add("warn", "The photo is hard to read. Retake it flat, in even light.")

    worst = max((RANK[lv] for lv, _ in checks), default=0)
    return ["ok", "caution", "bad"][worst], checks, levels


# ------------------------------------------------------------ drawing + voice
def draw_boxes(image, items, levels):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    size = max(14, w // 40)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    for key, (ymin, xmin, ymax, xmax) in items.items():
        color = COLORS[levels.get(key, "ok")]
        rect = [xmin * w / 1000, ymin * h / 1000, xmax * w / 1000, ymax * h / 1000]
        draw.rectangle(rect, outline=color, width=max(3, w // 200))
        label = LABELS[key]
        tw = draw.textlength(label, font=font)
        top = max(0, rect[1] - size - 6)
        draw.rectangle([rect[0], top, rect[0] + tw + 10, top + size + 6], fill=color)
        draw.text((rect[0] + 5, top + 2), label, fill="white", font=font)
    return img


def build_speech(code, d, verdict, df):
    s, months = SCRIPTS[code], MONTH_NAMES[code]
    parts = [s["intro"]]
    if not is_missing(d.get("brand_name")):
        parts.append(s["brand"].format(brand=d["brand_name"]))
    if not is_missing(d.get("composition")):
        parts.append(s["composition"].format(composition=str(d["composition"])[:120]))
    if not is_missing(d.get("batch_no")):
        parts.append(s["batch"].format(batch=d["batch_no"]))
    if not is_missing(d.get("manufacturer")):
        parts.append(s["manufacturer"].format(manufacturer=d["manufacturer"]))
    else:
        parts.append(s["manufacturer_missing"])

    exp = parse_month_year(d.get("expiry_date"))
    if exp:
        today = datetime.date.today()
        left = (exp[0] - today.year) * 12 + (exp[1] - today.month)
        key = "expired" if left < 0 else "expiry_soon" if left <= 3 else "expiry_ok"
        parts.append(s[key].format(month=months[exp[1] - 1], year=exp[0]))
    else:
        parts.append(s["expiry_unknown"])

    if is_missing(d.get("batch_no")):
        parts.append(s["batch_unknown"])
    else:
        key = re.sub(r"[^A-Z0-9]", "", str(d["batch_no"]).upper())
        flagged = not df[df["batch_key"] == key].empty
        parts.append(s["batch_flagged"] if flagged else s["batch_ok"])

    if d.get("spelling_mistakes"):
        parts.append(s["spelling_bad"])
    if d.get("visible_problems"):
        parts.append(s["photo_caution"])
    parts.append(s["final_" + verdict])
    return " ".join(parts)


@st.cache_data(show_spinner=False)
def make_audio(text, code):
    buf = io.BytesIO()
    gTTS(text=text, lang=code).write_to_fp(buf)
    return buf.getvalue()


# ----------------------------------------------------------------------- UI
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<p class="mm-title">💊 Medicine Mirror</p>', unsafe_allow_html=True)
st.markdown('<p class="mm-sub">Check a medicine strip in seconds.</p>', unsafe_allow_html=True)

if not get_api_key():
    st.error("This site is not set up yet: the reading service key is missing.")
    st.stop()

uploads = st.file_uploader(
    f"Add 1 to {MAX_PHOTOS} photos of the strip (front, back, or both)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True)

with st.expander("How to take a good photo"):
    st.markdown("- Lay the strip flat and fill the frame with it.\n"
                "- Use soft, even light and turn off the flash.\n"
                "- Tilt it slightly if the foil shows glare or shadows.\n"
                "- One photo is enough. Add a second photo of the other side to read more details.")

if not uploads:
    st.markdown(
        '<ul class="mm-points">'
        "<li>Reads the batch number, dates and manufacturer printed on the strip.</li>"
        "<li>Checks the expiry date and the drug regulator's flagged-batch list.</li>"
        "<li>Tells you the result out loud in English, Telugu or Hindi.</li>"
        "</ul>", unsafe_allow_html=True)
else:
    uploads = uploads[:MAX_PHOTOS]
    raw = [u.getvalue() for u in uploads]
    sig = tuple(hashlib.md5(b).hexdigest() for b in raw)
    if not allow_scan(sig):
        st.warning("The free daily limit has been reached. Please try again tomorrow.")
        st.stop()

    images, reads = [], []
    for n, b in enumerate(raw, 1):
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert("RGB")
        try:
            with st.spinner(f"Reading photo {n} of {len(raw)}..."):
                data = read_with_ai(image_to_bytes(image))
        except Exception as e:
            st.error("The reading service is busy or the photo could not be read. "
                     "Please wait a minute and try again.")
            with st.expander("Technical details"):
                st.write(str(e))
            st.stop()
        images.append(image)
        reads.append(data)

    df = load_data()
    merged, boxes = merge_reads(reads)
    verdict, checks, levels = check_strip(merged, df)
    for msg in find_conflicts(reads):
        checks.append(("warn", msg))
    verdict = ["ok", "caution", "bad"][max(RANK[lv] for lv, _ in checks)]

    # verdict card
    titles = {"bad": "Do not use this medicine",
              "caution": "Be careful: some details could not be read clearly",
              "ok": "No warning signs found"}
    if verdict == "bad":
        why = [m for lv, m in checks if lv == "bad"]
    elif verdict == "caution":
        warns = [m for lv, m in checks if lv == "warn"]
        why = warns[:4] + ([f"And {len(warns) - 4} more. See the checks below."] if len(warns) > 4 else [])
    else:
        why = ["This is a screening check, not proof that the medicine is genuine."]
    items = "".join(f"<li>{html.escape(m)}</li>" for m in why)
    st.markdown(f'<div class="verdict {verdict}"><div class="label">{titles[verdict]}</div>'
                f"<ul>{items}</ul></div>", unsafe_allow_html=True)

    missing = [LABELS[k] for k in ("brand_name", "composition", "manufacturer", "mfg_date")
               if is_missing(merged.get(k))]
    if missing:
        st.info("Not found yet: " + ", ".join(missing) +
                ". Optional: add another photo showing that part of the strip.")

    # voice
    st.subheader("Listen to the result")
    lang_name = st.radio("Voice language", list(LANGS), horizontal=True)
    code = LANGS[lang_name]
    speech = build_speech(code, merged, verdict, df)
    try:
        st.audio(make_audio(speech, code), format="audio/mp3")
    except Exception:
        st.warning("The voice needs an internet connection. The text is shown below.")
    st.caption(speech)

    # photos with highlights
    st.subheader("Your photos")
    for i, image in enumerate(images):
        trusted = reads[i].get("_source") == "gemini" or TRUST_OPENROUTER_BOXES
        own = {k: box for k, (idx, box) in boxes.items() if idx == i} if trusted else {}
        note = "" if own else " (highlights are not available for this photo)"
        st.image(draw_boxes(image, own, levels), caption=f"Photo {i + 1}{note}", width="stretch")

    # what was read
    st.subheader("What we read")
    rows = ""
    for key, label in LABELS.items():
        value = merged.get(key)
        if is_missing(value):
            rows += f'<div class="k">{label}</div><div class="v none">not visible</div>'
        else:
            rows += f'<div class="k">{label}</div><div class="v">{html.escape(str(value))}</div>'
    st.markdown(f'<div class="kv">{rows}</div>', unsafe_allow_html=True)

    # checks
    st.subheader("Checks")
    icons = {"ok": "✅", "warn": "🟠", "bad": "❌"}
    lis = "".join(f"<li><span>{icons[lv]}</span><span>{html.escape(m)}</span></li>" for lv, m in checks)
    st.markdown(f'<ul class="checks">{lis}</ul>', unsafe_allow_html=True)

    with st.expander("Technical details"):
        for i, d in enumerate(reads, 1):
            st.write(f"Photo {i}")
            st.json(d)

# ------------------------------------------------------------------- footer
try:
    months_loaded = ", ".join(sorted(set(load_data()["month"])))
    n_rows = len(load_data())
except Exception:
    months_loaded, n_rows = "not loaded", 0
st.markdown(
    '<div class="mm-foot"><b>Please read.</b> Medicine Mirror is a screening aid, not a laboratory test, '
    "and it cannot confirm that a medicine is genuine. If anything looks wrong, show the strip to a "
    "pharmacist or doctor. It reads only what is printed on the strip and compares the batch number with "
    f"{n_rows} batches flagged as Not of Standard Quality in the drug regulator's alerts ({html.escape(months_loaded)}), "
    "a small sample of all batches on the market.<br><br>"
    "<b>Privacy.</b> Your photos are sent to AI services (OpenRouter and Google Gemini) to read the text. "
    "Free AI services may keep or use what they receive. We do not store your photos. Please do not upload "
    "prescriptions or anything with a person's name or details.</div>", unsafe_allow_html=True)

if LINKEDIN_URL.startswith("https://"):
    st.markdown(
        '<div class="mm-devwrap"><a class="mm-dev" href="' + html.escape(LINKEDIN_URL, quote=True) +
        '" target="_blank" rel="noopener noreferrer">Developed by ' + html.escape(DEVELOPER_NAME) +
        "</a></div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="mm-devwrap">Developed by ' + html.escape(DEVELOPER_NAME) + "</div>",
                unsafe_allow_html=True)
